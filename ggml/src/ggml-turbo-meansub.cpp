#include "ggml-turbo-meansub.h"

#include <string.h>
#include <stdlib.h>

// Registry entry shape consumed by the generated data table. Compact storage: only live layers
// are stored ([n_live][max_c]) plus their layer indices; the accessor expands to a dense
// [max_l * max_c] slab (dead layers zero) on first use.
typedef struct {
    const char *  arch;
    int           n_layer;
    int           n_embd;
    int           max_l;
    int           max_c;
    int           k_live;
    int           v_live;
    const int *   klay;   // [k_live] layer indices
    const float * kval;   // [k_live * max_c] flattened
    const int *   vlay;   // [v_live]
    const float * vval;   // [v_live * max_c]
} ggml_meansub_entry;

#include "ggml-turbo-meansub-data.inc"   // defines g_meansub_table[] + g_meansub_count

static char g_arch[64] = {0};
static int  g_n_layer  = 0;
static int  g_n_embd   = 0;

static const ggml_meansub_entry * g_active = NULL;
static float * g_dense_k = NULL;
static float * g_dense_v = NULL;

GGML_API void ggml_turbo_meansub_set_model(const char * arch, int n_layer, int n_embd) {
    if (arch) {
        strncpy(g_arch, arch, sizeof(g_arch) - 1);
        g_arch[sizeof(g_arch) - 1] = 0;
    } else {
        g_arch[0] = 0;
    }
    g_n_layer = n_layer;
    g_n_embd  = n_embd;

    g_active = NULL;
    free(g_dense_k); g_dense_k = NULL;
    free(g_dense_v); g_dense_v = NULL;

    if (!g_arch[0]) {
        return;
    }
    for (int i = 0; i < g_meansub_count; i++) {
        const ggml_meansub_entry * e = &g_meansub_table[i];
        if (n_layer == e->n_layer && n_embd == e->n_embd && strcmp(g_arch, e->arch) == 0) {
            g_active = e;
            return;
        }
    }
}

static float * expand_dense(const ggml_meansub_entry * e, int kvsel) {
    const int     L      = e->max_l;
    const int     C      = e->max_c;
    const int     nlive  = kvsel ? e->v_live : e->k_live;
    const int *   lay    = kvsel ? e->vlay   : e->klay;
    const float * val    = kvsel ? e->vval   : e->kval;

    float * dense = (float *) calloc((size_t) L * C, sizeof(float));
    if (!dense) {
        return NULL;
    }
    for (int i = 0; i < nlive; i++) {
        const int l = lay[i];
        if (l < 0 || l >= L) {
            continue;
        }
        memcpy(dense + (size_t) l * C, val + (size_t) i * C, (size_t) C * sizeof(float));
    }
    return dense;
}

GGML_API const float * ggml_turbo_meansub_active(int kvsel, int * out_max_l, int * out_max_c, int * out_live) {
    if (!g_active) {
        return NULL;
    }
    const ggml_meansub_entry * e = g_active;
    if (out_max_l) *out_max_l = e->max_l;
    if (out_max_c) *out_max_c = e->max_c;
    if (out_live)  *out_live  = kvsel ? e->v_live : e->k_live;

    if (kvsel) {
        if (!g_dense_v) g_dense_v = expand_dense(e, 1);
        return g_dense_v;
    }
    if (!g_dense_k) g_dense_k = expand_dense(e, 0);
    return g_dense_k;
}
