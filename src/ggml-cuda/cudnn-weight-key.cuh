#pragma once

// Stable-identity key for the cuDNN reordered-conv-weight caches
// (conv2d-cudnn.cu g_weight_cache, conv3d-cudnn.cu g_weight3d_cache / g_weight3d_f32_cache).
//
// ★ WHY THIS EXISTS.
// Those caches used to be keyed by the weight's DEVICE POINTER. An address is only an
// identity for as long as the weight stays put, and under weight offload it does not:
// the model manager stages params into a temporary compute-backend buffer per graph and
// frees it afterwards (ModelManager::free_compute_staging_block), so the next render
// re-allocates the SAME address range with a DIFFERENT tensor -> address mapping. Every
// conv after the first render then took a STALE HIT and convolved with another tensor's
// reordered weights. Silent, seed-stable, and plausible-looking: job 0 correct, jobs 1+
// wrong. (Measured on krea2 edit mode: no adapter -> magenta/green noise; with the
// identity adapter -> a clean image of the wrong face.)
//
// A pointer is therefore the wrong identity. The ggml tensor NAME is the right one:
//   * it is what the model manager itself uses as the unique identity of a parameter
//     (ModelManager::tensor_states_by_name_ is a name->state map, and
//     resolve_required_tensor_states() hard-errors on an unnamed graph input), so it is
//     exactly as unique as the engine's own registry;
//   * staging swaps ->data/->buffer/->extra and leaves ->name alone, so it survives;
//   * it is the same every render, so the reorder now STAYS cached instead of being
//     dropped and rebuilt -- strictly cheaper than invalidating on every unstage.
//
// Correctness no longer depends on anybody remembering to invalidate when an address is
// recycled, which is the discipline that failed here (the conv2d release function was
// never wired to anything at all).
//
// ★ WHAT THE NAME DOES NOT COVER: the weight's CONTENT changing under a stable name.
// Two cases, both rare, both central and explicit:
//   1. a LoRA epoch change -- the delta is merged into the params copy
//      (ModelManager::fold_loras_into_params) or the staged copy
//      (apply_loras_to_params), so `first_stage_model...conv.weight` means something new;
//   2. a different checkpoint loaded into the same registered names.
// Those call ggml_backend_cuda_release_cudnn_conv{2,3}d_weights() from the site that
// changes the content -- NOT from the per-render hot path.
//
// The key also carries the shape, the ggml type, the backend buffer name and the CUDA
// device, so a same-named tensor of a different shape/dtype, or the same name on another
// device, cannot alias.
//
// GGML_CUDNN_WEIGHT_PTRKEY=1 restores the old device-pointer key. It exists to A/B this
// change; it is NOT a supported configuration -- under weight offload it reproduces the
// corruption described above.
// GGML_CUDNN_WEIGHT_TRACE=1 logs every reorder build and every hit whose source address
// moved (i.e. every hit the old pointer key would have missed or, worse, mis-hit).

#include "common.cuh"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

struct cudnn_weight_id {
    std::string  name;              // ggml tensor name; empty => keyed by ptr (see below)
    std::string  buf;               // backend buffer name, e.g. "CUDA0"
    const void * ptr    = nullptr;  // only populated for the ptr-key / unnamed fallback
    int64_t      ne[4]  = {0, 0, 0, 0};
    int32_t      type   = 0;
    int32_t      device = 0;

    bool operator==(const cudnn_weight_id & o) const {
        return ptr == o.ptr && type == o.type && device == o.device &&
               ne[0] == o.ne[0] && ne[1] == o.ne[1] && ne[2] == o.ne[2] && ne[3] == o.ne[3] &&
               name == o.name && buf == o.buf;
    }
};

struct cudnn_weight_id_hash {
    size_t operator()(const cudnn_weight_id & k) const {
        size_t h = std::hash<std::string>()(k.name);
        h = h * 1000003 ^ std::hash<std::string>()(k.buf);
        h = h * 1000003 ^ std::hash<const void *>()(k.ptr);
        for (int i = 0; i < 4; i++) {
            h = h * 1000003 ^ std::hash<int64_t>()(k.ne[i]);
        }
        h = h * 1000003 ^ std::hash<int32_t>()(k.type);
        h = h * 1000003 ^ std::hash<int32_t>()(k.device);
        return h;
    }
};

static inline bool cudnn_weight_ptrkey() {
    static const int on = [] {
        const char * e = getenv("GGML_CUDNN_WEIGHT_PTRKEY");
        return (e != nullptr && atoi(e) != 0) ? 1 : 0;
    }();
    return on != 0;
}

static inline bool cudnn_weight_trace() {
    static const int on = [] {
        const char * e = getenv("GGML_CUDNN_WEIGHT_TRACE");
        return (e != nullptr && atoi(e) != 0) ? 1 : 0;
    }();
    return on != 0;
}

static inline cudnn_weight_id cudnn_make_weight_id(const ggml_tensor * t, int device) {
    cudnn_weight_id k;

    if (cudnn_weight_ptrkey()) {
        // Escape hatch: reproduce the OLD key EXACTLY -- the raw device address and nothing
        // else. Deliberately not "pointer plus shape": carrying the shape would silently
        // turn most recycled-address collisions into misses, and then this lever would no
        // longer reproduce the corruption it exists to A/B against. This arm is expected to
        // render wrong at job index > 0 under weight offload. That is the point of it.
        k.ptr = t->data;
        return k;
    }

    k.ne[0]  = t->ne[0];
    k.ne[1]  = t->ne[1];
    k.ne[2]  = t->ne[2];
    k.ne[3]  = t->ne[3];
    k.type   = (int32_t) t->type;
    k.device = (int32_t) device;

    const char * nm = t->name;
    if (nm == nullptr || nm[0] == '\0') {
        // No stable identity available (an unnamed graph temporary): fall back to the device
        // pointer, still qualified by shape/type/device so a recycled address holding a
        // differently-shaped tensor cannot stale-hit. Unnamed weights are not expected here
        // -- every parameter that reaches a conv is a registered, named tensor (or a named
        // view of one) -- so warn once, because it is the one shape in which this cache is
        // still partly address-keyed.
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            fprintf(stderr,
                    "[cudnn-weight-cache] UNNAMED conv weight [%lld,%lld,%lld,%lld] type=%d -- "
                    "falling back to the device-pointer key, which is NOT stable under weight "
                    "offload. Name the tensor if this render is wrong at job index > 0.\n",
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                    (long long) t->ne[3], (int) t->type);
        }
        k.ptr = t->data;
    } else {
        k.name = nm;
        if (t->buffer != nullptr) {
            const char * bn = ggml_backend_buffer_name(t->buffer);
            if (bn != nullptr) {
                k.buf = bn;
            }
        }
    }
    return k;
}

// Shared trace helper: `tag` is the cache ("conv2d" / "conv3d" / "conv3d-f32"),
// `what` is "build" or "moved".
static inline void cudnn_weight_trace_log(const char * tag, const char * what,
                                          const cudnn_weight_id & k, const void * src) {
    fprintf(stderr, "[cudnn-weight-cache] %-10s %-5s name='%s' buf='%s' ne=[%lld,%lld,%lld,%lld] "
                    "type=%d dev=%d src=%p\n",
            tag, what, k.name.empty() ? "<unnamed>" : k.name.c_str(), k.buf.c_str(),
            (long long) k.ne[0], (long long) k.ne[1], (long long) k.ne[2], (long long) k.ne[3],
            (int) k.type, (int) k.device, src);
}
