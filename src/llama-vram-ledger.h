#pragma once

// VRAM demand ledger — substrate for VBR co-tenancy auto-negotiation (design v3.8).
//
// Fork processes coordinate GPU memory through tiny per-(device,pid) files in a per-uid
// tmpfs directory: a loader that cannot allocate publishes a demand claim; residents read
// it and may shed KV pages (protocol logic lives with the consumers, not here). This TU
// owns only the substrate: directory arming, the on-disk format, the write discipline,
// peer liveness, scanning/GC, and the process-singleton writer thread that heartbeats
// every live claim (never the tick or decode boundaries — an idle-gated beat would
// false-stall a donor's lift detector).
//
// Write discipline: field changes go through atomic tmp+rename (they bump dir mtime and
// so trip readers' cheap pre-checks); heartbeats and bytes_now go through in-place
// pwrites of a seqlock'd slot (file mtime only — a lone resident must not trip its own
// pre-check at token rate). Linux-complete; elsewhere llama_vram_ledger_armed() returns
// false and every operation degrades to the clean-fail path.

#include <cstdint>
#include <string>
#include <vector>

// base constants (design spec v3.8) — everything else is a derivation so retunes cannot
// desync the family
constexpr int64_t LLAMA_VRAM_LEDGER_LONG_MS       = 45000;
constexpr int64_t LLAMA_VRAM_LEDGER_SHORT_MS      =  8000;
constexpr int64_t LLAMA_VRAM_LEDGER_BEAT_MS       =  2000;
constexpr int64_t LLAMA_VRAM_LEDGER_IDLE_MS       = 30000;
constexpr int64_t LLAMA_VRAM_LEDGER_HEADROOM_BASE = 192ll*1024*1024; // per live process, per device
constexpr int64_t LLAMA_VRAM_LEDGER_DEBOUNCE      = 10;              // scan events
constexpr int64_t LLAMA_VRAM_LEDGER_SCAN_CAP      = 64;              // files per scan (warn-once beyond)
// derived
constexpr int64_t LLAMA_VRAM_LEDGER_REMOVE_MS         = 3*LLAMA_VRAM_LEDGER_LONG_MS; // claim GC; > extension cap by construction
constexpr int64_t LLAMA_VRAM_LEDGER_RUNTIME_IGNORE_MS = 5*LLAMA_VRAM_LEDGER_BEAT_MS;
constexpr int64_t LLAMA_VRAM_LEDGER_HB_STALL_MS       = 3*LLAMA_VRAM_LEDGER_BEAT_MS;
constexpr int64_t LLAMA_VRAM_LEDGER_EXT_CAP_MS        = 2*LLAMA_VRAM_LEDGER_LONG_MS + 2*LLAMA_VRAM_LEDGER_BEAT_MS;
// nominal ask for the compute-buffer reserve hold (sched sizes are internal); derived
// from HEADROOM_BASE so a headroom retune cannot silently make nominal asks unserviceable
constexpr int64_t LLAMA_VRAM_LEDGER_NOMINAL_ASK       = LLAMA_VRAM_LEDGER_HEADROOM_BASE + 64ll*1024*1024;

// the one headroom number the protocol's two halves must agree on: HEADROOM_BASE with
// the VBR_VRAM_HEADROOM_MIB env override applied (donor shortfall math AND demander
// READY/sufficiency both call this)
size_t llama_vram_headroom_bytes();

enum llama_vram_claim_phase : uint32_t {
    LLAMA_VRAM_CLAIM_PROBE     = 0,
    LLAMA_VRAM_CLAIM_DEMAND    = 1,
    LLAMA_VRAM_CLAIM_SATISFIED = 2,
    LLAMA_VRAM_CLAIM_RUNTIME   = 3,
};

// rename-written claim fields (the CRC-covered body; heartbeat/bytes_now live in the
// seqlock slot and are set through their own calls)
struct llama_vram_claim_fields {
    llama_vram_claim_phase phase;
    uint64_t bytes_total_remaining_est;
    uint32_t est_partial;    // 1 = no plan hint, model-remaining-only estimate
    uint64_t ver;            // per-process demand-generation counter
    uint64_t created_ts_ns;  // fixed once per demand ver, identical across its per-device files
};

// one validated peer file as read by a scan; heartbeat/bytes_now are raw snapshots —
// staleness aging against previous sightings is the consumer's state, not the TU's
struct llama_vram_peer_claim {
    std::string busid;
    int32_t     pid;
    uint64_t    starttime;
    llama_vram_claim_fields fields;
    uint64_t    hb_counter;
    uint64_t    bytes_now;
};

// true when a tmpfs-backed per-uid ledger directory is available (created 0700 on first
// call); false disables the whole feature — callers fail clean with SHORT patience
bool llama_vram_ledger_armed();
const std::string & llama_vram_ledger_dir();

// self/peer identity (starttime = /proc/<pid>/stat f22; liveness pairs pid with it so a
// recycled pid never impersonates a dead claimant)
int32_t  llama_vram_ledger_self_pid(); // tie-break input; 0 when unarmed
uint64_t llama_vram_ledger_self_starttime();
uint64_t llama_vram_ledger_now_ns(); // CLOCK_BOOTTIME (advances across suspend, unlike
                                     // ggml_time's CLOCK_MONOTONIC) — created_ts domain
bool     llama_vram_ledger_pid_alive(int32_t pid, uint64_t starttime);

// own-claim lifecycle. publish rename-writes ggml-vram-claim-<busid>-<pid> and registers
// it with the writer thread (lazily started, exits when no claim remains); set_bytes
// mutates the registry — the writer thread lands it in the seqlock slot within one beat;
// withdraw unlinks and unregisters. All are no-ops returning false/void when not armed.
bool llama_vram_claim_publish  (const std::string & busid, const llama_vram_claim_fields & fields);
void llama_vram_claim_set_bytes(const std::string & busid, uint64_t bytes_now);
bool llama_vram_claim_withdraw (const std::string & busid);
void llama_vram_claim_withdraw_all();

// scan every claim file (SCAN_CAP-capped), validate (magic/version/CRC/sane ranges;
// malformed → ignored), and GC files whose owner is dead (pid gone or starttime
// mismatch) — GC authority is any scanner. Own files are skipped. Returns count in out.
int llama_vram_ledger_scan(std::vector<llama_vram_peer_claim> & out);

// ---- presence markers: ggml-vram-resident-<busid>-<pid> ----
// Written by every fork process holding device memory. shed_available/grant_pending are
// the donor-protocol fields. Marker heartbeat freshness measures RESPONSIVENESS (a deaf
// resident cannot act), so beats ride the owner's SCAN cadence — llama_vram_marker_beat
// is the scan-event hook (internally rate-limited to one per BEAT), never a thread.

struct llama_vram_marker_fields {
    uint32_t vbr;            // 1 = dynamic-VBR resident (potential donor)
    uint32_t serviced;       // 1 = runs an idle tick (llama-server); 0 = boundary-scans only
    uint64_t shed_available; // bytes the f16->t8 band could free on this device (0 = no offer)
    uint64_t grant_pending;  // granted-but-not-yet-flushed bytes on this device
};

// one validated peer marker as read by a scan
struct llama_vram_peer_marker {
    std::string busid;
    int32_t     pid;
    uint64_t    starttime;
    uint64_t    created_ts_ns; // fixed at the marker's FIRST write (donor-rank input)
    llama_vram_marker_fields fields;
    uint64_t    hb_counter;
};

// rename-write the marker (field change — bumps dir mtime; the first write fixes
// created_ts for the marker's lifetime). Returns false when unarmed or on I/O failure.
bool llama_vram_marker_publish(const std::string & busid, const llama_vram_marker_fields & fields);
// true if this process already published a marker for the busid — context init uses it
// to publish a vbr:0 presence marker only when the VBR side has not (and will not be
// downgraded by a later non-VBR context in the same process, e.g. a draft model)
bool llama_vram_marker_present(const std::string & busid);
// in-place heartbeat, called from the owner's scan events; no-op if never published or
// if beaten less than BEAT ago
void llama_vram_marker_beat(const std::string & busid);
bool llama_vram_marker_withdraw(const std::string & busid);
void llama_vram_marker_withdraw_all();

// scan peer markers (own skipped; dead-owner GC; SCAN_CAP applies)
int llama_vram_ledger_scan_markers(std::vector<llama_vram_peer_marker> & out);

// process-wide serviced flag: set once by a host that runs an idle tick (llama-server);
// consumed by marker publication. Default false (cli/perplexity/bench are deaf at the
// prompt and must not upgrade a demander's patience window).
void llama_vram_marker_set_serviced(bool serviced);
bool llama_vram_marker_serviced_flag();

// ledger directory mtime (ns, 0 when unavailable) — the ~1µs per-boundary pre-check:
// rename-writes bump it, in-place beats do not
uint64_t llama_vram_ledger_dir_mtime_ns();
