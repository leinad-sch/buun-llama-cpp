#include "llama-vram-ledger.h"

#include "llama-impl.h"

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// on-disk layout, fixed offsets (the seqlock slot is pwritten in place and must not move):
//   0  u32 magic  4 u32 version  8 u32 body_crc  12 u32 reserved
//   body (CRC-covered, rename-written):
//   16 u32 phase  20 u32 est_partial  24 i32 pid  28 u32 reserved
//   32 u64 starttime  40 u64 ver  48 u64 bytes_total_remaining_est  56 u64 created_ts_ns
//   seqlock slot (in-place, excluded from body_crc, valid iff seq even and stable):
//   64 u32 hb_seq  68 u32 reserved  72 u64 hb_counter  80 u64 bytes_now
static constexpr uint32_t VRLG_MAGIC     = 0x474C5256; // "VRLG"
static constexpr uint32_t VRLG_VERSION   = 1;
static constexpr size_t   VRLG_FILE_SIZE = 88;
static constexpr size_t   VRLG_BODY_OFF  = 16;
static constexpr size_t   VRLG_SLOT_OFF  = 64;

static constexpr const char * VRLG_CLAIM_PREFIX = "ggml-vram-claim-";

size_t llama_vram_headroom_bytes() {
    static const size_t headroom = [] {
        const char * e = getenv("VBR_VRAM_HEADROOM_MIB");
        return e != nullptr ? (size_t) strtoull(e, nullptr, 10) * 1024 * 1024
                            : (size_t) LLAMA_VRAM_LEDGER_HEADROOM_BASE;
    }();
    return headroom;
}

// presence markers share the 88-byte layout and slot offsets; their body is:
//   16 u32 vbr  20 u32 serviced  24 i32 pid  28 u32 reserved
//   32 u64 starttime  40 u64 created_ts_ns  48 u64 shed_available  56 u64 grant_pending
static constexpr uint32_t     VRLM_MAGIC         = 0x4D4C5256; // "VRLM"
static constexpr const char * VRLM_MARKER_PREFIX = "ggml-vram-resident-";

#ifdef __linux__

static void vrlg_put_u32(uint8_t * buf, size_t off, uint32_t v) { memcpy(buf + off, &v, 4); }
static void vrlg_put_u64(uint8_t * buf, size_t off, uint64_t v) { memcpy(buf + off, &v, 8); }
static uint32_t vrlg_get_u32(const uint8_t * buf, size_t off) { uint32_t v; memcpy(&v, buf + off, 4); return v; }
static uint64_t vrlg_get_u64(const uint8_t * buf, size_t off) { uint64_t v; memcpy(&v, buf + off, 8); return v; }

// ---- arming: per-uid 0700 dir on a tmpfs-like filesystem ----

static bool vrlg_is_tmpfs(const char * path) {
    struct statfs sfs;
    if (statfs(path, &sfs) != 0) {
        return false;
    }
    return sfs.f_type == 0x01021994 /*TMPFS*/ || sfs.f_type == 0x858458F6 /*RAMFS*/;
}

static std::string vrlg_resolve_dir() {
    const char * base = getenv("XDG_RUNTIME_DIR");
    if (base == nullptr || !vrlg_is_tmpfs(base)) {
        base = "/dev/shm";
        if (!vrlg_is_tmpfs(base)) {
            return "";
        }
    }
    std::string dir = std::string(base) + "/ggml-vram-ledger-" + std::to_string(getuid());
    if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
        return "";
    }
    // refuse a hijacked path: must be our own non-symlink 0700 directory
    struct stat st;
    if (lstat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != getuid()) {
        return "";
    }
    if ((st.st_mode & 0777) != 0700 && chmod(dir.c_str(), 0700) != 0) {
        return "";
    }
    return dir;
}

static const std::string & vrlg_dir() {
    static const std::string dir = vrlg_resolve_dir();
    return dir;
}

// ---- identity and liveness ----

// starttime = field 22 of /proc/<pid>/stat; comm (field 2) may contain spaces, so parse
// from the last ')'
static uint64_t vrlg_pid_starttime(int32_t pid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return 0;
    }
    char buf[1024];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    const char * p = strrchr(buf, ')');
    if (p == nullptr) {
        return 0;
    }
    p++; // now at field 3 (state); starttime is field 22 -> skip 18 more fields
    uint64_t starttime = 0;
    int field = 3;
    while (*p != '\0') {
        while (*p == ' ') p++;
        if (field == 22) {
            starttime = strtoull(p, nullptr, 10);
            break;
        }
        while (*p != '\0' && *p != ' ') p++;
        field++;
    }
    return starttime;
}

// ---- own-claim registry + writer thread (process singleton) ----

struct vrlg_claim_entry {
    std::string path;
    int         fd = -1;   // kept open for slot pwrites
    uint32_t    hb_seq    = 0;
    uint64_t    bytes_now = 0;
};

struct vrlg_state {
    std::mutex mu;
    std::condition_variable cv;
    std::unordered_map<std::string, vrlg_claim_entry> claims; // key: busid
    std::thread th;
    bool th_running = false;

    ~vrlg_state() {
        {
            std::lock_guard<std::mutex> lock(mu);
            for (auto & [busid, e] : claims) {
                if (e.fd >= 0) {
                    close(e.fd);
                }
                unlink(e.path.c_str()); // leftover files would only burden peers' GC
            }
            claims.clear();
            cv.notify_all(); // empty registry -> the writer thread exits
        }
        if (th.joinable()) {
            th.join();
        }
    }
};

static vrlg_state & vrlg() {
    static vrlg_state s;
    return s;
}

// seqlock write: odd seq -> payload -> even seq; readers discard torn slots. The beat
// counter is hb_seq/2 by construction (both advance only here), so it is not stored.
static void vrlg_beat_entry(vrlg_claim_entry & e) {
    uint8_t slot[VRLG_FILE_SIZE - VRLG_SLOT_OFF];
    e.hb_seq += 1; // odd: write in progress
    vrlg_put_u32(slot, 0, e.hb_seq);
    pwrite(e.fd, slot, 4, VRLG_SLOT_OFF);
    e.hb_seq += 1; // even: the value the stable slot will carry
    vrlg_put_u32(slot, 4, 0);
    vrlg_put_u64(slot, 8, e.hb_seq/2);
    vrlg_put_u64(slot, 16, e.bytes_now);
    pwrite(e.fd, slot + 4, sizeof(slot) - 4, VRLG_SLOT_OFF + 4);
    vrlg_put_u32(slot, 0, e.hb_seq);
    pwrite(e.fd, slot, 4, VRLG_SLOT_OFF);
}

static void vrlg_thread_main() {
    auto & s = vrlg();
    std::unique_lock<std::mutex> lock(s.mu);
    while (true) {
        if (s.claims.empty()) {
            s.th_running = false; // exits when no live claim remains; publish restarts
            return;
        }
        for (auto & [busid, e] : s.claims) {
            vrlg_beat_entry(e);
        }
        // half of BEAT keeps every consumer comfortably under the <= BEAT cadence
        s.cv.wait_for(lock, std::chrono::milliseconds(LLAMA_VRAM_LEDGER_BEAT_MS/2));
    }
}

// caller holds s.mu
static void vrlg_ensure_thread(vrlg_state & s) {
    if (s.th_running) {
        return;
    }
    if (s.th.joinable()) {
        s.th.join();
    }
    s.th_running = true;
    s.th = std::thread(vrlg_thread_main);
}

static std::string vrlg_claim_path(const std::string & busid, int32_t pid) {
    return vrlg_dir() + "/" + VRLG_CLAIM_PREFIX + busid + "-" + std::to_string(pid);
}

// parse+validate one peer claim file. Returns: 1 = valid (pc filled), 0 = ignore
// (malformed/torn — never trusted, never GC'd here), -1 = owner dead (caller GCs).
static int vrlg_read_claim_file(const std::string & path, int32_t pid, llama_vram_peer_claim & pc) {
    uint8_t buf[VRLG_FILE_SIZE];
    int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (fd < 0) {
        return 0;
    }
    // seqlock read of the slot region: retry while a beat is mid-write
    bool stable = false;
    for (int attempt = 0; attempt < 3 && !stable; ++attempt) {
        if (pread(fd, buf, sizeof(buf), 0) != (ssize_t) sizeof(buf)) {
            break;
        }
        uint32_t seq1 = vrlg_get_u32(buf, VRLG_SLOT_OFF);
        if (seq1 % 2 != 0) {
            continue;
        }
        uint8_t slot2[4];
        if (pread(fd, slot2, 4, VRLG_SLOT_OFF) != 4 || vrlg_get_u32(slot2, 0) != seq1) {
            continue;
        }
        stable = true;
    }
    close(fd);
    if (!stable ||
        vrlg_get_u32(buf, 0) != VRLG_MAGIC ||
        vrlg_get_u32(buf, 4) != VRLG_VERSION ||
        vrlg_get_u32(buf, 8) != llama_crc32(buf + VRLG_BODY_OFF, VRLG_SLOT_OFF - VRLG_BODY_OFF) ||
        (int32_t) vrlg_get_u32(buf, 24) != pid ||
        vrlg_get_u32(buf, 16) > LLAMA_VRAM_CLAIM_RUNTIME) {
        return 0;
    }
    uint64_t starttime = vrlg_get_u64(buf, 32);
    if (!llama_vram_ledger_pid_alive(pid, starttime)) {
        return -1;
    }
    pc.pid        = pid;
    pc.starttime  = starttime;
    pc.fields.phase       = (llama_vram_claim_phase) vrlg_get_u32(buf, 16);
    pc.fields.est_partial = vrlg_get_u32(buf, 20);
    pc.fields.ver         = vrlg_get_u64(buf, 40);
    pc.fields.bytes_total_remaining_est = vrlg_get_u64(buf, 48);
    pc.fields.created_ts_ns             = vrlg_get_u64(buf, 56);
    pc.hb_counter = vrlg_get_u64(buf, VRLG_SLOT_OFF +  8);
    pc.bytes_now  = vrlg_get_u64(buf, VRLG_SLOT_OFF + 16);
    return 1;
}

// ---- public API ----

bool llama_vram_ledger_armed() {
    return !vrlg_dir().empty();
}

const std::string & llama_vram_ledger_dir() {
    return vrlg_dir();
}

int32_t llama_vram_ledger_self_pid() {
    return (int32_t) getpid();
}

uint64_t llama_vram_ledger_self_starttime() {
    static const uint64_t st = vrlg_pid_starttime(getpid());
    return st;
}

uint64_t llama_vram_ledger_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return (uint64_t) ts.tv_sec*1000000000ull + (uint64_t) ts.tv_nsec;
}

bool llama_vram_ledger_pid_alive(int32_t pid, uint64_t starttime) {
    if (kill(pid, 0) != 0 && errno == ESRCH) {
        return false;
    }
    return vrlg_pid_starttime(pid) == starttime;
}

bool llama_vram_claim_publish(const std::string & busid, const llama_vram_claim_fields & fields) {
    if (!llama_vram_ledger_armed()) {
        return false;
    }
    auto & s = vrlg();
    std::lock_guard<std::mutex> lock(s.mu);

    // heartbeat continuity across field-change republishes of the same claim
    const auto prev = s.claims.find(busid);
    const uint32_t hb_seq    = prev != s.claims.end() ? prev->second.hb_seq    : 0;
    const uint64_t bytes_now = prev != s.claims.end() ? prev->second.bytes_now : 0;

    uint8_t buf[VRLG_FILE_SIZE] = {0};
    vrlg_put_u32(buf,  0, VRLG_MAGIC);
    vrlg_put_u32(buf,  4, VRLG_VERSION);
    vrlg_put_u32(buf, 16, (uint32_t) fields.phase);
    vrlg_put_u32(buf, 20, fields.est_partial);
    vrlg_put_u32(buf, 24, (uint32_t) getpid());
    vrlg_put_u64(buf, 32, llama_vram_ledger_self_starttime());
    vrlg_put_u64(buf, 40, fields.ver);
    vrlg_put_u64(buf, 48, fields.bytes_total_remaining_est);
    vrlg_put_u64(buf, 56, fields.created_ts_ns);
    vrlg_put_u32(buf,  8, llama_crc32(buf + VRLG_BODY_OFF, VRLG_SLOT_OFF - VRLG_BODY_OFF));
    vrlg_put_u32(buf, VRLG_SLOT_OFF +  0, hb_seq);
    vrlg_put_u64(buf, VRLG_SLOT_OFF +  8, hb_seq/2);
    vrlg_put_u64(buf, VRLG_SLOT_OFF + 16, bytes_now);

    // all file work happens on locals; the registry is only touched once everything
    // succeeded, so no failure path needs cleanup beyond its own file descriptors.
    // atomic tmp+rename: field changes must bump dir mtime for readers' pre-checks
    std::string tmp = vrlg_dir() + "/.claim-tmp-XXXXXX";
    int tmp_fd = mkstemp(tmp.data());
    if (tmp_fd < 0) {
        return false;
    }
    bool ok = write(tmp_fd, buf, sizeof(buf)) == (ssize_t) sizeof(buf);
    close(tmp_fd);
    std::string path = vrlg_claim_path(busid, getpid());
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        return false;
    }
    int fd = open(path.c_str(), O_WRONLY | O_NOFOLLOW);
    if (fd < 0) {
        unlink(path.c_str());
        return false;
    }

    auto & e = s.claims[busid];
    if (e.fd >= 0) {
        close(e.fd);
    }
    e.path      = path;
    e.fd        = fd;
    e.hb_seq    = hb_seq;
    e.bytes_now = bytes_now;
    vrlg_ensure_thread(s);
    s.cv.notify_all();
    return true;
}

void llama_vram_claim_set_bytes(const std::string & busid, uint64_t bytes_now) {
    auto & s = vrlg();
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.claims.find(busid);
    if (it == s.claims.end()) {
        return;
    }
    // no wakeup: the scheduled beat lands this within one BEAT/2, well inside the
    // <= BEAT contract — waking the writer per call would thrash at token rate
    it->second.bytes_now = bytes_now;
}

bool llama_vram_claim_withdraw(const std::string & busid) {
    auto & s = vrlg();
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.claims.find(busid);
    if (it == s.claims.end()) {
        return false;
    }
    if (it->second.fd >= 0) {
        close(it->second.fd);
    }
    unlink(it->second.path.c_str());
    s.claims.erase(it);
    s.cv.notify_all();
    return true;
}

void llama_vram_claim_withdraw_all() {
    auto & s = vrlg();
    std::lock_guard<std::mutex> lock(s.mu);
    for (auto & [busid, e] : s.claims) {
        if (e.fd >= 0) {
            close(e.fd);
        }
        unlink(e.path.c_str());
    }
    s.claims.clear();
    s.cv.notify_all();
}

int llama_vram_ledger_scan(std::vector<llama_vram_peer_claim> & out) {
    out.clear();
    if (!llama_vram_ledger_armed()) {
        return 0;
    }
    DIR * d = opendir(vrlg_dir().c_str());
    if (d == nullptr) {
        return 0;
    }
    static bool warned_cap = false;
    int seen = 0;
    const size_t prefix_len = strlen(VRLG_CLAIM_PREFIX);
    for (struct dirent * de = readdir(d); de != nullptr; de = readdir(d)) {
        if (strncmp(de->d_name, VRLG_CLAIM_PREFIX, prefix_len) != 0) {
            continue;
        }
        if (++seen > LLAMA_VRAM_LEDGER_SCAN_CAP) {
            if (!warned_cap) {
                LLAMA_LOG_WARN("%s: more than %d ledger files, ignoring the excess\n",
                        __func__, (int) LLAMA_VRAM_LEDGER_SCAN_CAP);
                warned_cap = true;
            }
            break;
        }
        // <busid>-<pid>: pid is the suffix after the last '-'
        const char * name = de->d_name + prefix_len;
        const char * dash = strrchr(name, '-');
        if (dash == nullptr || dash == name) {
            continue;
        }
        int32_t pid = (int32_t) atoi(dash + 1);
        if (pid <= 0 || pid == getpid()) {
            continue;
        }
        std::string path = vrlg_dir() + "/" + de->d_name;
        llama_vram_peer_claim pc;
        switch (vrlg_read_claim_file(path, pid, pc)) {
            case 1:
                pc.busid = std::string(name, dash - name);
                out.push_back(std::move(pc));
                break;
            case -1:
                unlink(path.c_str()); // GC authority: any scanner
                break;
            default:
                break;
        }
    }
    closedir(d);
    return (int) out.size();
}

uint64_t llama_vram_ledger_dir_mtime_ns() {
    if (!llama_vram_ledger_armed()) {
        return 0;
    }
    struct stat st;
    if (stat(vrlg_dir().c_str(), &st) != 0) {
        return 0;
    }
    return (uint64_t) st.st_mtim.tv_sec*1000000000ull + (uint64_t) st.st_mtim.tv_nsec;
}

// ---- presence markers ----
// no writer thread: publish is a rename (field change), beats are caller-driven in-place
// pwrites on the owner's scan cadence with a one-per-BEAT internal rate limit

static bool g_vrlm_serviced = false;

void llama_vram_marker_set_serviced(bool serviced) {
    g_vrlm_serviced = serviced;
}

bool llama_vram_marker_serviced_flag() {
    return g_vrlm_serviced;
}

struct vrlm_marker_entry {
    std::string path;
    int         fd = -1;
    uint32_t    hb_seq        = 0;
    uint64_t    created_ts_ns = 0; // fixed at first publish, survives republishes
    uint64_t    last_beat_ns  = 0;
};

// markers are only touched from the owner's decode/tick thread — no lock needed beyond
// the map's own lifetime (static storage, destroyed at exit after unlinking)
struct vrlm_state {
    std::unordered_map<std::string, vrlm_marker_entry> markers; // key: busid

    ~vrlm_state() {
        for (auto & [busid, e] : markers) {
            if (e.fd >= 0) {
                close(e.fd);
            }
            unlink(e.path.c_str());
        }
    }
};

static vrlm_state & vrlm() {
    static vrlm_state s;
    return s;
}

bool llama_vram_marker_publish(const std::string & busid, const llama_vram_marker_fields & fields) {
    if (!llama_vram_ledger_armed()) {
        return false;
    }
    auto & s = vrlm();
    const auto prev = s.markers.find(busid);
    const uint32_t hb_seq        = prev != s.markers.end() ? prev->second.hb_seq        : 0;
    const uint64_t created_ts_ns = prev != s.markers.end() ? prev->second.created_ts_ns
                                                           : llama_vram_ledger_now_ns();

    uint8_t buf[VRLG_FILE_SIZE] = {0};
    vrlg_put_u32(buf,  0, VRLM_MAGIC);
    vrlg_put_u32(buf,  4, VRLG_VERSION);
    vrlg_put_u32(buf, 16, fields.vbr);
    vrlg_put_u32(buf, 20, fields.serviced);
    vrlg_put_u32(buf, 24, (uint32_t) getpid());
    vrlg_put_u64(buf, 32, llama_vram_ledger_self_starttime());
    vrlg_put_u64(buf, 40, created_ts_ns);
    vrlg_put_u64(buf, 48, fields.shed_available);
    vrlg_put_u64(buf, 56, fields.grant_pending);
    vrlg_put_u32(buf,  8, llama_crc32(buf + VRLG_BODY_OFF, VRLG_SLOT_OFF - VRLG_BODY_OFF));
    vrlg_put_u32(buf, VRLG_SLOT_OFF + 0, hb_seq);
    vrlg_put_u64(buf, VRLG_SLOT_OFF + 8, hb_seq/2);

    std::string tmp = vrlg_dir() + "/.marker-tmp-XXXXXX";
    int tmp_fd = mkstemp(tmp.data());
    if (tmp_fd < 0) {
        return false;
    }
    bool ok = write(tmp_fd, buf, sizeof(buf)) == (ssize_t) sizeof(buf);
    close(tmp_fd);
    std::string path = vrlg_dir() + "/" + VRLM_MARKER_PREFIX + busid + "-" + std::to_string(getpid());
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        return false;
    }
    int fd = open(path.c_str(), O_WRONLY | O_NOFOLLOW);
    if (fd < 0) {
        unlink(path.c_str());
        return false;
    }
    auto & e = s.markers[busid];
    if (e.fd >= 0) {
        close(e.fd);
    }
    e.path          = path;
    e.fd            = fd;
    e.hb_seq        = hb_seq;
    e.created_ts_ns = created_ts_ns;
    return true;
}

void llama_vram_marker_beat(const std::string & busid) {
    auto & s = vrlm();
    auto it = s.markers.find(busid);
    if (it == s.markers.end() || it->second.fd < 0) {
        return;
    }
    auto & e = it->second;
    const uint64_t now = llama_vram_ledger_now_ns();
    if (now - e.last_beat_ns < (uint64_t) LLAMA_VRAM_LEDGER_BEAT_MS * 1000000ull) {
        return;
    }
    e.last_beat_ns = now;
    uint8_t slot[VRLG_FILE_SIZE - VRLG_SLOT_OFF];
    e.hb_seq += 1;
    vrlg_put_u32(slot, 0, e.hb_seq);
    pwrite(e.fd, slot, 4, VRLG_SLOT_OFF);
    e.hb_seq += 1;
    vrlg_put_u32(slot, 4, 0);
    vrlg_put_u64(slot, 8, e.hb_seq/2);
    vrlg_put_u64(slot, 16, 0);
    pwrite(e.fd, slot + 4, sizeof(slot) - 4, VRLG_SLOT_OFF + 4);
    vrlg_put_u32(slot, 0, e.hb_seq);
    pwrite(e.fd, slot, 4, VRLG_SLOT_OFF);
}

bool llama_vram_marker_withdraw(const std::string & busid) {
    auto & s = vrlm();
    auto it = s.markers.find(busid);
    if (it == s.markers.end()) {
        return false;
    }
    if (it->second.fd >= 0) {
        close(it->second.fd);
    }
    unlink(it->second.path.c_str());
    s.markers.erase(it);
    return true;
}

void llama_vram_marker_withdraw_all() {
    auto & s = vrlm();
    for (auto & [busid, e] : s.markers) {
        if (e.fd >= 0) {
            close(e.fd);
        }
        unlink(e.path.c_str());
    }
    s.markers.clear();
}

int llama_vram_ledger_scan_markers(std::vector<llama_vram_peer_marker> & out) {
    out.clear();
    if (!llama_vram_ledger_armed()) {
        return 0;
    }
    DIR * d = opendir(vrlg_dir().c_str());
    if (d == nullptr) {
        return 0;
    }
    static bool warned_cap = false;
    int seen = 0;
    const size_t prefix_len = strlen(VRLM_MARKER_PREFIX);
    for (struct dirent * de = readdir(d); de != nullptr; de = readdir(d)) {
        if (strncmp(de->d_name, VRLM_MARKER_PREFIX, prefix_len) != 0) {
            continue;
        }
        if (++seen > LLAMA_VRAM_LEDGER_SCAN_CAP) {
            if (!warned_cap) {
                LLAMA_LOG_WARN("%s: more than %d marker files, ignoring the excess\n",
                        __func__, (int) LLAMA_VRAM_LEDGER_SCAN_CAP);
                warned_cap = true;
            }
            break;
        }
        const char * name = de->d_name + prefix_len;
        const char * dash = strrchr(name, '-');
        if (dash == nullptr || dash == name) {
            continue;
        }
        int32_t pid = (int32_t) atoi(dash + 1);
        if (pid <= 0 || pid == getpid()) {
            continue;
        }
        std::string path = vrlg_dir() + "/" + de->d_name;

        uint8_t buf[VRLG_FILE_SIZE];
        int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW);
        if (fd < 0) {
            continue;
        }
        bool stable = false;
        for (int attempt = 0; attempt < 3 && !stable; ++attempt) {
            if (pread(fd, buf, sizeof(buf), 0) != (ssize_t) sizeof(buf)) {
                break;
            }
            uint32_t seq1 = vrlg_get_u32(buf, VRLG_SLOT_OFF);
            if (seq1 % 2 != 0) {
                continue;
            }
            uint8_t slot2[4];
            if (pread(fd, slot2, 4, VRLG_SLOT_OFF) != 4 || vrlg_get_u32(slot2, 0) != seq1) {
                continue;
            }
            stable = true;
        }
        close(fd);
        if (!stable ||
            vrlg_get_u32(buf, 0) != VRLM_MAGIC ||
            vrlg_get_u32(buf, 4) != VRLG_VERSION ||
            vrlg_get_u32(buf, 8) != llama_crc32(buf + VRLG_BODY_OFF, VRLG_SLOT_OFF - VRLG_BODY_OFF) ||
            (int32_t) vrlg_get_u32(buf, 24) != pid ||
            vrlg_get_u32(buf, 16) > 1 || vrlg_get_u32(buf, 20) > 1) {
            continue;
        }
        uint64_t starttime = vrlg_get_u64(buf, 32);
        if (!llama_vram_ledger_pid_alive(pid, starttime)) {
            unlink(path.c_str()); // GC authority: any scanner
            continue;
        }
        llama_vram_peer_marker pm;
        pm.busid         = std::string(name, dash - name);
        pm.pid           = pid;
        pm.starttime     = starttime;
        pm.created_ts_ns = vrlg_get_u64(buf, 40);
        pm.fields.vbr            = vrlg_get_u32(buf, 16);
        pm.fields.serviced       = vrlg_get_u32(buf, 20);
        pm.fields.shed_available = vrlg_get_u64(buf, 48);
        pm.fields.grant_pending  = vrlg_get_u64(buf, 56);
        pm.hb_counter    = vrlg_get_u64(buf, VRLG_SLOT_OFF + 8);
        out.push_back(std::move(pm));
    }
    closedir(d);
    return (int) out.size();
}

#else // !__linux__ — substrate disabled: armed() = false, every caller takes the clean-fail path

bool llama_vram_ledger_armed() { return false; }

const std::string & llama_vram_ledger_dir() {
    static const std::string empty;
    return empty;
}

int32_t  llama_vram_ledger_self_pid()       { return 0; }
uint64_t llama_vram_ledger_self_starttime() { return 0; }
uint64_t llama_vram_ledger_now_ns()         { return 0; }

bool llama_vram_ledger_pid_alive(int32_t, uint64_t) { return false; }

bool llama_vram_claim_publish  (const std::string &, const llama_vram_claim_fields &) { return false; }
void llama_vram_claim_set_bytes(const std::string &, uint64_t) {}
bool llama_vram_claim_withdraw (const std::string &) { return false; }
void llama_vram_claim_withdraw_all() {}

int llama_vram_ledger_scan(std::vector<llama_vram_peer_claim> & out) { out.clear(); return 0; }

uint64_t llama_vram_ledger_dir_mtime_ns() { return 0; }

bool llama_vram_marker_publish(const std::string &, const llama_vram_marker_fields &) { return false; }
void llama_vram_marker_beat(const std::string &) {}
bool llama_vram_marker_withdraw(const std::string &) { return false; }
void llama_vram_marker_withdraw_all() {}

int llama_vram_ledger_scan_markers(std::vector<llama_vram_peer_marker> & out) { out.clear(); return 0; }

static bool g_vrlm_serviced_stub = false;
void llama_vram_marker_set_serviced(bool serviced) { g_vrlm_serviced_stub = serviced; }
bool llama_vram_marker_serviced_flag() { return g_vrlm_serviced_stub; }

#endif
