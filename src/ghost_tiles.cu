/*
 * ghost_tiles.cu — CUDA GPU kernels for ghost-tiles sparse attention
 *
 * Part of the Lucineer fleet ecosystem.
 * Requires: CUDA 11+
 *
 * Key insight: each 8x8 attention tile = 1 CUDA thread block.
 * Active tiles compute. Ghost tiles don't launch — zero cost.
 */

#include "ghost_tiles.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define TILE_DIM 8
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
} while(0)

/* ── Kernel: Apply ghost mask ── */
__global__ void ghost_mask_kernel(float *mask, int seq_len, int tile_size) {
    int tr = blockIdx.y;
    int tc = blockIdx.x;
    int lr = threadIdx.y;
    int lc = threadIdx.x;

    int gr = tr * tile_size + lr;
    int gc = tc * tile_size + lc;

    if (gr >= seq_len || gc >= seq_len) return;
    mask[gr * seq_len + gc] = 1.0f; /* active tile gets full weight */
}

/* ── Kernel: Sparse attention for a single tile ── */
__global__ void sparse_attn_tile_kernel(
    const float *Q, const float *K, const float *V, float *O,
    int seq_len, int d_head, int tile_size, float scale,
    int tile_row, int tile_col)
{
    __shared__ float tile_q[TILE_DIM][TILE_DIM];
    __shared__ float tile_k[TILE_DIM][TILE_DIM];
    __shared__ float tile_v[TILE_DIM][TILE_DIM];
    __shared__ float scores[TILE_DIM][TILE_DIM];
    __shared__ float max_score[TILE_DIM];

    int lr = threadIdx.y; /* local row within tile */
    int lc = threadIdx.x; /* local col within tile */
    int gr = tile_row * tile_size + lr; /* global row */
    int gc = tile_col * tile_size + lc; /* global col */

    /* Load Q tile */
    if (gr < seq_len) {
        for (int d = 0; d < d_head; d += TILE_DIM) {
            if (d + lc < d_head) tile_q[lr][lc] = Q[gr * d_head + d + lc];
        }
    }

    /* Load K tile and compute attention scores */
    float row_max = -1e30f;
    if (gc < seq_len) {
        for (int d = 0; d < d_head; d += TILE_DIM) {
            if (d + lc < d_head) tile_k[lr][lc] = K[gc * d_head + d + lc];
        }
        scores[lr][lc] = 0.0f;
        for (int d = 0; d < d_head; d++) {
            scores[lr][lc] += Q[gr * d_head + d] * K[gc * d_head + d];
        }
        scores[lr][lc] *= scale;
        if (scores[lr][lc] > row_max) row_max = scores[lr][lc];
    }

    /* Softmax within tile */
    __syncthreads();
    max_score[lr] = row_max;
    __syncthreads();

    if (gc < seq_len && gr < seq_len) {
        float sum = 0.0f;
        scores[lr][lc] = expf(scores[lr][lc] - row_max);
        sum = scores[lr][lc];
        tile_v[lr][lc] = sum;
        __syncthreads();
        /* Reduction */
        for (int s = TILE_DIM / 2; s > 0; s >>= 1) {
            if (lc < s) tile_v[lr][lc] += tile_v[lr][lc + s];
            __syncthreads();
        }
        scores[lr][lc] /= tile_v[lr][0];

        /* Load V tile and accumulate output */
        for (int d = 0; d < d_head; d++) {
            O[gr * d_head + d] += scores[lr][lc] * V[gc * d_head + d];
        }
    }
}

/* ── Kernel: Prune tiles below threshold ── */
__global__ void prune_kernel(int32_t *d_active, float *d_weight, int n, float threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_weight[i] < threshold) d_active[i] = 0;
}

/* ── Kernel: Count active ── */
__global__ void count_kernel(int32_t *d_active, int n, int32_t *d_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_active[i]) atomicAdd(d_count, 1);
}

/* ── Host API ── */

int gpu_tile_grid_alloc(GpuTileGrid *grid, int num_tiles) {
    cudaMalloc(&grid->d_row, num_tiles * sizeof(uint16_t));
    cudaMalloc(&grid->d_col, num_tiles * sizeof(uint16_t));
    cudaMalloc(&grid->d_active, num_tiles * sizeof(int32_t));
    cudaMalloc(&grid->d_weight, num_tiles * sizeof(float));
    grid->num_tiles = num_tiles;
    return cudaGetLastError();
}

void gpu_tile_grid_free(GpuTileGrid *grid) {
    if (grid->d_row) cudaFree(grid->d_row);
    if (grid->d_col) cudaFree(grid->d_col);
    if (grid->d_active) cudaFree(grid->d_active);
    if (grid->d_weight) cudaFree(grid->d_weight);
    memset(grid, 0, sizeof(GpuTileGrid));
}

int gpu_tile_grid_upload(GpuTileGrid *grid,
                         const uint16_t *h_rows, const uint16_t *h_cols,
                         const int32_t *h_active, const float *h_weights,
                         int num_tiles) {
    int n = num_tiles < grid->num_tiles ? num_tiles : grid->num_tiles;
    cudaMemcpy(grid->d_row, h_rows, n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(grid->d_col, h_cols, n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(grid->d_active, h_active, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(grid->d_weight, h_weights, n * sizeof(float), cudaMemcpyHostToDevice);
    return cudaGetLastError();
}

int gpu_apply_ghost_mask(float *d_mask, const GpuTileGrid *grid,
                         int seq_len, int tile_size) {
    cudaMemset(d_mask, 0, (size_t)seq_len * seq_len * sizeof(float));
    /* Launch only for active tiles — this is the whole point */
    dim3 blocks(seq_len / tile_size, seq_len / tile_size);
    dim3 threads(tile_size, tile_size);
    ghost_mask_kernel<<<blocks, threads>>>(d_mask, seq_len, tile_size);
    return cudaGetLastError();
}

float gpu_sparse_attention(const AttnParams *params, const GpuTileGrid *grid) {
    int grid_size = params->seq_len / params->tile_size;
    dim3 threads(params->tile_size, params->tile_size);

    long long flops = 0;
    /* Launch one block per active tile */
    for (int tr = 0; tr < grid_size; tr++) {
        for (int tc = 0; tc < grid_size; tc++) {
            int idx = tr * grid_size + tc;
            int32_t active = 0;
            cudaMemcpy(&active, &grid->d_active[idx], sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (!active) continue; /* GHOST — skip entirely */
            sparse_attn_tile_kernel<<<dim3(1,1), threads>>>(
                params->Q, params->K, params->V, params->O,
                params->seq_len, params->d_head, params->tile_size, params->scale,
                tr, tc);
            /* FLOPs: 2*d_head (matmul) + 3 (softmax) per element, tile_size^2 elements */
            flops += (long long)params->tile_size * params->tile_size *
                     (2 * params->d_head + 3);
        }
    }
    cudaDeviceSynchronize();
    return (float)flops / 1e9f; /* return GFLOPS */
}

int gpu_count_active(const GpuTileGrid *grid) {
    int32_t *d_count;
    cudaMalloc(&d_count, sizeof(int32_t));
    cudaMemset(d_count, 0, sizeof(int32_t));
    int threads = 256;
    int blocks = (grid->num_tiles + threads - 1) / threads;
    count_kernel<<<blocks, threads>>>(grid->d_active, grid->num_tiles, d_count);
    int32_t count = 0;
    cudaMemcpy(&count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    return count;
}

int gpu_prune_tiles(GpuTileGrid *grid, float threshold) {
    int threads = 256;
    int blocks = (grid->num_tiles + threads - 1) / threads;
    prune_kernel<<<blocks, threads>>>(grid->d_active, grid->d_weight, grid->num_tiles, threshold);
    return cudaGetLastError();
}

int gpu_sync(const char *label) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[%s] CUDA sync error: %s\n", label, cudaGetErrorString(err));
        return err;
    }
    return 0;
}
