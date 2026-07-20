#include "llama-vram-demand.h"

#include "llama-impl.h"
#include "llama-vram-ledger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

// All times ns on the CLOCK_BOOTTIME domain (llama_vram_ledger_now_ns).
static constexpr uint64_t NS_PER_MS = 1000000ull;

namespace {

struct dev_state {
    ggml_backend_dev_t dev = nullptr;
    uint64_t planned     = 0;     // plan hint, or seeded from the first failure
    uint64_t landed      = 0;     // bytes successfully allocated so far (== published bytes_now)
    bool     est_partial = false; // planned was seeded from a failure, not a plan hint
    uint32_t revisions   = 0;     // upward est revisions consumed (spec: exactly one honored)
    bool     demanded    = false; // a failure occurred (or plan remainder > 0) this attempt
};

// observation aging for peer heartbeats: freshness is measured by the READER's clock
// across unchanged counter reads — a first sighting counts as a beat
struct hb_obs {
    uint64_t counter   = 0;
    uint64_t change_ns = 0;
};

struct demander {
    // process-wide plan hints (device_id/busid -> bytes), set before load
    std::map<std::string, uint64_t> plan;

    // a concluded attempt failed for good (insufficiency / expiry / lost tie-break):
    // every later alloc failure in the SAME load fails fast — patience windows must not
    // stack across the loader's (and the fit's) many buffer allocations. Cleared when a
    // load concludes (satisfied or abandoned), so the next load gets a fresh attempt.
    bool terminal_failed = false;

    // live attempt
    bool     attempt_open  = false;
    bool     committed     = false; // probe -> demand happened
    bool     satisfied     = false;
    uint64_t ver           = 0;
    uint64_t created_ts    = 0;
    uint64_t deadline      = 0;     // patience deadline (extensible, capped)
    bool     long_window   = false; // sticky upgrade happened
    std::map<std::string, dev_state> devs;      // busid -> state
    std::map<std::string, hb_obs>    marker_obs; // "busid-pid" -> aging
    std::map<std::string, uint64_t>  claim_progress; // peer claim key -> last bytes_now (tie-break)
    std::map<std::string, llama_vram_claim_fields> last_pub; // publish memo (rename on change only)
    uint64_t last_free_seen = 0;    // post-commit progress signal (primary demanded dev)
};

demander & dm() {
    static demander d;
    return d;
}

std::string dev_busid(ggml_backend_dev_t dev) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.device_id != nullptr ? std::string(props.device_id) : std::string();
}

uint64_t dev_free(ggml_backend_dev_t dev) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    return (uint64_t) free;
}

uint64_t est_remaining(const dev_state & ds) {
    if (ds.planned > ds.landed) {
        return ds.planned - ds.landed;
    }
    return 0;
}

// age a peer heartbeat against our observation history; returns age in ns
uint64_t observe_hb(demander & d, const std::string & key, uint64_t counter, uint64_t now) {
    auto & o = d.marker_obs[key];
    if (o.change_ns == 0 || o.counter != counter) {
        o.counter   = counter;
        o.change_ns = now;
        return 0;
    }
    return now - o.change_ns;
}

void unlink_all(demander & d, const char * reason) {
    if (d.attempt_open) {
        LLAMA_LOG_INFO("vram-demand: attempt ver %llu closed (%s)\n",
                (unsigned long long) d.ver, reason);
    }
    llama_vram_claim_withdraw_all();
    d.attempt_open = false;
    d.committed    = false;
    d.satisfied    = false;
    d.devs.clear();
    d.marker_obs.clear();
    d.claim_progress.clear();
    d.last_pub.clear();
}

// write/refresh the claim files for every demanded device at the given phase. Rename
// discipline: a publish with unchanged fields is skipped — a redundant rename would bump
// the dir mtime and knock every resident donor off its stable fast path per iteration.
bool publish_claims(demander & d, llama_vram_claim_phase phase) {
    bool any = false;
    for (auto & [busid, ds] : d.devs) {
        if (!ds.demanded) {
            continue;
        }
        llama_vram_claim_fields f = {};
        f.phase                     = phase;
        f.bytes_total_remaining_est = est_remaining(ds);
        // spec: the honored upward revision rewrites est and clears est_partial
        f.est_partial               = (ds.est_partial && ds.revisions == 0) ? 1u : 0u;
        f.ver                       = d.ver;
        f.created_ts_ns             = d.created_ts;
        auto last = d.last_pub.find(busid);
        if (last == d.last_pub.end() ||
            last->second.phase != f.phase ||
            last->second.bytes_total_remaining_est != f.bytes_total_remaining_est ||
            last->second.est_partial != f.est_partial) {
            any = llama_vram_claim_publish(busid, f) || any;
            d.last_pub[busid] = f;
        }
        llama_vram_claim_set_bytes(busid, ds.landed); // in-place, never a rename
    }
    return any;
}

} // namespace

void llama_vram_plan_hint_set(const char * device_id, uint64_t bytes) {
    if (device_id == nullptr) {
        return;
    }
    dm().plan[device_id] = bytes;
}

ggml_backend_buffer_t llama_vram_hold_alloc_ctx_tensors(ggml_context * ctx,
                                                        ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return buf;
    }
    if (buf == nullptr) {
        size_t need = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            need += ggml_backend_buft_get_alloc_size(buft, t);
        }
        while (buf == nullptr && llama_vram_demand_hold(dev, need)) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
    }
    if (buf != nullptr) {
        llama_vram_demand_alloc_landed(dev, ggml_backend_buffer_get_size(buf));
    }
    return buf;
}

void llama_vram_demand_alloc_landed(ggml_backend_dev_t dev, size_t bytes) {
    auto & d = dm();
    if (!d.attempt_open) {
        return;
    }
    const std::string busid = dev_busid(dev);
    auto it = d.devs.find(busid);
    if (it == d.devs.end()) {
        return;
    }
    it->second.landed += bytes;
    llama_vram_claim_set_bytes(busid, it->second.landed);
}

bool llama_vram_demand_hold(ggml_backend_dev_t dev, size_t bytes) {
    auto & d = dm();
    if (!llama_vram_ledger_armed()) {
        // no substrate: honor SHORT patience with plain backoff — the grace for a
        // resident that cannot be seen is pointless, but a clean bounded wait keeps
        // behavior uniform (design: clean fail + SHORT patience)
        static thread_local uint64_t bare_deadline = 0;
        const uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (bare_deadline == 0) {
            bare_deadline = now + (uint64_t) LLAMA_VRAM_LEDGER_SHORT_MS * NS_PER_MS;
        }
        if (now >= bare_deadline) {
            bare_deadline = 0;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(LLAMA_VRAM_LEDGER_BEAT_MS));
        return true;
    }

    const uint64_t now = llama_vram_ledger_now_ns();
    const std::string busid = dev_busid(dev);
    if (busid.empty()) {
        return false; // CPU/unidentified device: nothing to negotiate
    }
    if (d.terminal_failed) {
        return false; // this load already negotiated and lost — fail fast
    }

    // ---- attempt open (ABSENT -> PROBE) ----
    if (!d.attempt_open) {
        d.attempt_open = true;
        d.committed    = false;
        d.satisfied    = false;
        d.ver++;
        d.created_ts   = now;
        d.long_window  = false;
        d.deadline     = now + (uint64_t) LLAMA_VRAM_LEDGER_SHORT_MS * NS_PER_MS;
        d.devs.clear();
        d.marker_obs.clear();
        d.claim_progress.clear();
        // every device with a plan-hint remainder is part of the joint claim
        for (const auto & [hint_busid, hint_bytes] : d.plan) {
            dev_state ds;
            ds.planned  = hint_bytes;
            ds.demanded = false;
            d.devs.emplace(hint_busid, ds);
        }
    }
    auto & ds = d.devs[busid];
    ds.dev      = dev;
    ds.demanded = true;
    if (ds.planned == 0) {
        // no hint: current failure size only, flagged est_partial (one upward revision
        // may later be honored)
        ds.planned     = ds.landed + bytes;
        ds.est_partial = true;
    } else if (ds.landed + bytes > ds.planned) {
        // the estimate was short. An est_partial claim gets exactly ONE honored upward
        // revision (locally checkable: a third ask is refused); a plan-hinted claim's
        // shortfall means the hint lied — same single-revision grace.
        if (ds.revisions == 0) {
            ds.planned = ds.landed + bytes;
            ds.revisions = 1;
            LLAMA_LOG_INFO("vram-demand: est revision on %s -> %.1f MiB remaining\n",
                    busid.c_str(), est_remaining(ds)/1048576.0);
            if (d.committed) {
                // donors size their shed from the claim file — the revised est must land
                publish_claims(d, LLAMA_VRAM_CLAIM_DEMAND);
            }
        } else {
            LLAMA_LOG_WARN("vram-demand: %s needs %.1f MiB beyond the revised estimate — refusing a third ask\n",
                    busid.c_str(), (ds.landed + bytes - ds.planned)/1048576.0);
            unlink_all(d, "estimate exhausted");
            d.terminal_failed = true;
            return false;
        }
    }
    // demanded set may have grown — mark plan-remainder devices demanded too (joint claim)
    for (auto & [b, s] : d.devs) {
        if (est_remaining(s) > 0) {
            s.demanded = true;
        }
    }

    if (!d.committed) {
        publish_claims(d, LLAMA_VRAM_CLAIM_PROBE);
    }

    // ---- scan peers ----
    std::vector<llama_vram_peer_marker> markers;
    llama_vram_ledger_scan_markers(markers);
    std::vector<llama_vram_peer_claim> claims;
    llama_vram_ledger_scan(claims);

    const uint64_t window_ms   = d.long_window ? LLAMA_VRAM_LEDGER_LONG_MS : LLAMA_VRAM_LEDGER_SHORT_MS;
    const uint64_t fresh_ns    = (uint64_t) LLAMA_VRAM_LEDGER_LONG_MS/2 * NS_PER_MS; // freshness = LONG/2 for selection/upgrade
    const uint64_t offer_fresh = (uint64_t) window_ms/2 * NS_PER_MS;                 // offer counting uses the selected window
    const uint64_t cap         = d.created_ts + (uint64_t) LLAMA_VRAM_LEDGER_EXT_CAP_MS * NS_PER_MS;

    // ---- patience upgrade (sticky): a qualifying beat inside the active window ----
    bool progress_event = false;
    for (const auto & m : markers) {
        const std::string key = m.busid + "-" + std::to_string(m.pid);
        const uint64_t age = observe_hb(d, key, m.hb_counter, now);
        const bool demanded_dev = d.devs.count(m.busid) != 0 && d.devs[m.busid].demanded;
        if (!d.long_window && now < d.deadline &&
            m.fields.serviced == 1 && m.fields.vbr == 1 &&
            m.fields.shed_available > 0 && demanded_dev && age < fresh_ns) {
            d.long_window = true;
            d.deadline = std::min(cap, now + (uint64_t) LLAMA_VRAM_LEDGER_LONG_MS * NS_PER_MS);
            LLAMA_LOG_INFO("vram-demand: potential donor on %s (pid %d, offer %.1f MiB) — patience %llds\n",
                    m.busid.c_str(), m.pid, m.fields.shed_available/1048576.0,
                    (long long) LLAMA_VRAM_LEDGER_LONG_MS/1000);
        }
        // post-commit progress: grant_pending on a demanded device's donor marker
        if (d.committed && demanded_dev && m.fields.grant_pending > 0) {
            progress_event = true;
        }
    }

    // ---- pre-commit: tie-break against earlier claims, then READY/sufficiency ----
    if (!d.committed) {
        for (const auto & c : claims) {
            if (c.fields.created_ts_ns < d.created_ts ||
                (c.fields.created_ts_ns == d.created_ts && c.pid < llama_vram_ledger_self_pid())) {
                // an earlier live claimant: outwait within patience; if it PROGRESSES, we
                // fail fast rather than both grinding to expiry
                const std::string key = "claim-" + c.busid + "-" + std::to_string(c.pid);
                auto it = d.claim_progress.find(key);
                if (it != d.claim_progress.end() && c.bytes_now > it->second) {
                    LLAMA_LOG_WARN("vram-demand: earlier claimant pid %d is progressing — yielding\n", c.pid);
                    unlink_all(d, "lost tie-break");
                    d.terminal_failed = true;
                    return false;
                }
                d.claim_progress[key] = c.bytes_now;
                // outwait: skip READY evaluation this round
                goto backoff;
            }
        }
        {
            // READY per demanded device, single sufficiency evaluation at the trigger point
            bool all_ready = true;
            for (const auto & [b, s] : d.devs) {
                if (!s.demanded || s.dev == nullptr) {
                    continue;
                }
                const uint64_t free_b = dev_free(s.dev);
                const uint64_t need   = est_remaining(s);
                const bool offers_needed = free_b < need + llama_vram_headroom_bytes();
                if (!offers_needed) {
                    continue; // free covers -> READY
                }
                bool fresh_offer = false, any_marker = false, all_fresh = true;
                for (const auto & m : markers) {
                    if (m.busid != b) {
                        continue;
                    }
                    any_marker = true;
                    const uint64_t age = observe_hb(d, m.busid + "-" + std::to_string(m.pid), m.hb_counter, now);
                    if (age >= offer_fresh) {
                        all_fresh = false;
                    } else if (m.fields.shed_available > 0) {
                        fresh_offer = true;
                    }
                }
                // READY ⟺ free-covers ∨ ∃ fresh offer ∨ (markers nonempty ∧ all fresh)
                if (!fresh_offer && !(any_marker && all_fresh)) {
                    all_ready = false;
                }
            }
            if (all_ready || now >= d.deadline) {
                // joint sufficiency, evaluated ONCE
                bool sufficient = true;
                for (const auto & [b, s] : d.devs) {
                    if (!s.demanded || s.dev == nullptr) {
                        continue;
                    }
                    uint64_t offers = 0;
                    for (const auto & m : markers) {
                        if (m.busid == b && m.fields.shed_available > 0 &&
                            observe_hb(d, m.busid + "-" + std::to_string(m.pid), m.hb_counter, now) < offer_fresh) {
                            offers += m.fields.shed_available;
                        }
                    }
                    const uint64_t free_b = dev_free(s.dev);
                    const uint64_t headroom = llama_vram_headroom_bytes();
                    const uint64_t have   = offers + (free_b > headroom ? free_b - headroom : 0);
                    if (have < est_remaining(s)) {
                        LLAMA_LOG_WARN("vram-demand: device %s insufficient even with offers "
                                "(need %.1f MiB, offers %.1f MiB + free %.1f MiB)\n",
                                b.c_str(), est_remaining(s)/1048576.0, offers/1048576.0, free_b/1048576.0);
                        sufficient = false;
                    }
                }
                if (!sufficient) {
                    unlink_all(d, "insufficient");
                    d.terminal_failed = true;
                    return false;
                }
                publish_claims(d, LLAMA_VRAM_CLAIM_DEMAND);
                d.committed = true;
                d.last_free_seen = dev_free(dev);
                LLAMA_LOG_INFO("vram-demand: demand committed (ver %llu) — waiting for donors\n",
                        (unsigned long long) d.ver);
            }
        }
    } else {
        // ---- post-commit progress extension: free movement or own landed bytes ----
        const uint64_t free_b = dev_free(dev);
        if (free_b > d.last_free_seen) {
            progress_event = true;
        }
        d.last_free_seen = free_b;
    }

    if (progress_event && d.committed) {
        d.deadline = std::min(cap, now + (uint64_t) LLAMA_VRAM_LEDGER_LONG_MS * NS_PER_MS);
    }

    if (now >= d.deadline && (d.committed || now >= cap)) {
        // committed and out of patience (or hard cap): honest per-device failure
        for (const auto & [b, s] : d.devs) {
            if (s.demanded) {
                LLAMA_LOG_ERROR("vram-demand: needed %.1f MiB on %s, %.1f MiB free after %.1fs — giving up\n",
                        est_remaining(s)/1048576.0, b.c_str(),
                        s.dev ? dev_free(s.dev)/1048576.0 : 0.0,
                        (now - d.created_ts)/1e9);
            }
        }
        unlink_all(d, "patience expired");
        d.terminal_failed = true;
        return false;
    }

backoff:
    // backoff ∈ [BEAT, 2·BEAT], final step clamped to the deadline
    {
        uint64_t sleep_ms = (uint64_t) LLAMA_VRAM_LEDGER_BEAT_MS
                          + (d.ver + (now / NS_PER_MS)) % (uint64_t) LLAMA_VRAM_LEDGER_BEAT_MS;
        const uint64_t hard = d.committed ? d.deadline : cap;
        if (now + sleep_ms * NS_PER_MS > hard) {
            sleep_ms = std::max<uint64_t>(1, (hard - now) / NS_PER_MS);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
    return true;
}

void llama_vram_demand_satisfied() {
    auto & d = dm();
    d.terminal_failed = false; // load concluded — the next load negotiates fresh
    if (!d.attempt_open || !d.committed) {
        // an attempt that never committed (or never opened) has nothing to hold onto
        unlink_all(d, "load done without demand");
        return;
    }
    publish_claims(d, LLAMA_VRAM_CLAIM_SATISFIED);
    d.satisfied = true;
    LLAMA_LOG_INFO("vram-demand: satisfied (ver %llu) — claim held until first decode\n",
            (unsigned long long) d.ver);
}

void llama_vram_demand_abandon() {
    unlink_all(dm(), "abandoned");
    dm().terminal_failed = false; // load concluded — the next load negotiates fresh
}

void llama_vram_demand_complete() {
    auto & d = dm();
    if (!d.satisfied) {
        return;
    }
    unlink_all(d, "claim-complete");
}

bool llama_vram_demand_pending_complete() {
    return dm().satisfied;
}
