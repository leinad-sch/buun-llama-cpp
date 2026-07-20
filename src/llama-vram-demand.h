#pragma once

// Loader-patience demand protocol — the DEMANDER side of VBR co-tenancy (design v3.8).
//
// A loader that hits a device-allocation failure holds inside a bounded patience window
// instead of failing instantly: it publishes a claim in the VRAM ledger (probe), waits for
// resident donors' offers (presence markers), commits the demand once every demanded
// device is READY and jointly sufficient, and keeps retrying with backoff while donors
// shed. The call sites drive the loop:
//
//     while (!(buf = try_alloc()) && llama_vram_demand_hold(dev, bytes)) {}
//
// hold() returns true = retry now (one backoff step elapsed inside), false = give up
// (patience expired, insufficiency, or lost a tie-break) — the caller then fails exactly
// as it does today, with the honest terminal message already logged. The demander has no
// tick: scans, GC, tie-break and sufficiency all run inline in hold().
//
// Single-threaded by contract (model load / context init); never called on decode paths —
// the runtime fast-fail wall stays intact by construction.

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

// plan hint: total bytes this process still intends to allocate on the device (model +
// context + fit totals). Written by common_init_from_params before load; a demander with
// no hint publishes its current failure size only, flagged est_partial (donors may honor
// one upward revision). Keyed by the PCI bus id from ggml_backend_dev_props.device_id.
void llama_vram_plan_hint_set(const char * device_id, uint64_t bytes);

// bytes landed on the device (successful buffer alloc during a held load) — feeds the
// claim's self-measured bytes_now and the amortized grant release on the donor side
void llama_vram_demand_alloc_landed(ggml_backend_dev_t dev, size_t bytes);

// hold-aware ctx-tensor allocation for load-phase GPU buffers: first try, then (GPU
// devices only) hold within patience while donors shed, retrying; EVERY success reports
// its bytes as landed (first-try successes during a held load must feed bytes_now too).
// Returns nullptr when the hold gives up — callers fail exactly as before.
ggml_backend_buffer_t llama_vram_hold_alloc_ctx_tensors(struct ggml_context * ctx,
                                                        ggml_backend_buffer_type_t buft);

// hold after a failed alloc of `bytes` on `dev`. First call opens the attempt (probe);
// every call sleeps one backoff step. True = retry the alloc; false = attempt over
// (claims already unlinked, terminal reason logged).
bool llama_vram_demand_hold(ggml_backend_dev_t dev, size_t bytes);

// the load finished successfully with a live claim → flip it to phase=satisfied (the
// writer thread keeps beating; donors keep their grants armed until claim-complete)
void llama_vram_demand_satisfied();

// the load failed for good, or the process is exiting → unlink everything
void llama_vram_demand_abandon();

// claim-complete: first successful decode with n_outputs > 0 → unlink (donors lift on
// disappearance-with-live-pid). No-op when no claim is live — callable unconditionally.
void llama_vram_demand_complete();

// true while a satisfied claim awaits its first real decode (cheap inline gate for the
// decode-path complete hook)
bool llama_vram_demand_pending_complete();
