/*
 * ghost_tiles.cu — CUDA GPU kernels for ghost-tiles sparse attention
 *
 * Part of the Lucineer fleet ecosystem.
 * Requires: CUDA 11+, ghost-tiles-c (host library)
 *
 * Each tile maps to a GPU thread block. Active tiles compute attention;
 * ghost tiles skip entirely — zero FLOPs, zero memory traffic.
 *
 * Integration:
 *   - cuda-ghost-tiles (Rust): Host-side pattern management
 *   - flux-runtime-c (C): Opcode GHOST_TILE dispatches to these kernels
 *   - cuda-neural-compiler (Rust): METL format outputs ghost-tile configs
 */

#ifndef GHOST_TILES_CUH
#define GHOST_TILES_CUH

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── GPU tile descriptor (soa layout for coalesced access) ── */
typedef struct {
    uint16_t *d_row;       /* device ptr */
    uint16_t *d_col;
    int32_t  *d_active;    /* 0 or 1 */
    float    *d_weight;
    int       num_tiles;
} GpuTileGrid;

/* ── Attention tile kernel parameters ── */
typedef struct {
    const float *Q;        /* Query matrix [seq_len x d_head] */
    const float *K;        /* Key matrix   [seq_len x d_head] */
    const float *V;        /* Value matrix [seq_len x d_head] */
    float       *O;        /* Output matrix [seq_len x d_head] */
    float       *mask;     /* Ghost tile mask [seq_len x seq_len] */
    int          seq_len;
    int          d_head;
    int          tile_size;
    float        scale;     /* 1/sqrt(d_head) */
} AttnParams;

/* ── CUDA API ── */

/* Allocate GPU tile grid */
int gpu_tile_grid_alloc(GpuTileGrid *grid, int num_tiles);

/* Free GPU tile grid */
void gpu_tile_grid_free(GpuTileGrid *grid);

/* Copy active tiles from host to GPU. Returns number of active tiles copied. */
int gpu_tile_grid_upload(GpuTileGrid *grid,
                         const uint16_t *h_rows,
                         const uint16_t *h_cols,
                         const int32_t  *h_active,
                         const float    *h_weights,
                         int num_tiles);

/* Apply ghost tile mask to full attention matrix.
 * mask[i*seq_len + j] = tile_weight if active, 0.0 if ghost.
 * Launches: ceil(seq_len/tile_size)^2 thread blocks, tile_size x tile_size threads each. */
int gpu_apply_ghost_mask(float *d_mask,
                         const GpuTileGrid *grid,
                         int seq_len, int tile_size);

/* Sparse attention kernel: compute only active tiles.
 * Q, K, V are [seq_len x d_head] row-major.
 * Output O = softmax(QK^T/sqrt(d)) @ V for active tiles only.
 *
 * Ghost tiles contribute zero — skipped entirely.
 *
 * Returns: GFLOPS actually computed (for energy accounting). */
float gpu_sparse_attention(const AttnParams *params, const GpuTileGrid *grid);

/* Count active tiles on GPU */
int gpu_count_active(const GpuTileGrid *grid);

/* Prune tiles on GPU (set active=0 for tiles below threshold) */
int gpu_prune_tiles(GpuTileGrid *grid, float weight_threshold);

/* Synchronize and check for CUDA errors */
int gpu_sync(const char *label);

#ifdef __cplusplus
}
#endif

#endif /* GHOST_TILES_CUH */
