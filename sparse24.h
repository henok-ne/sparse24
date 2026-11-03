/* sparse24.h -- C ABI for a 2:4 structured-sparse FP32 GEMM on ARM64 NEON.
 *
 * Dense mathematical operation (general entry point):
 *     C[M, N] = W_pruned[M, K] * B[K, N]
 *
 * W is stored in 2:4 structured-sparse form:
 *   - K is a multiple of 4.
 *   - Each row of W is split into K/4 groups of 4 consecutive weights.
 *   - Each group keeps exactly two coordinate slots (the two largest |w|).
 *   - packed_weights layout, row-major:
 *         M rows, each row has K/2 floats:
 *         [g0_slot0, g0_slot1, g1_slot0, g1_slot1, ...]
 *   - indices layout, row-major:
 *         M rows, each row has K/4 uint8_t tokens:
 *         bits 0..3 = original column offset (0..3) of slot0
 *         bits 4..7 = original column offset (0..3) of slot1
 *         slot0 coordinate < slot1 coordinate.
 *
 * B is row-major K x N. C is row-major M x N.
 *
 * ABI note:
 *   A general GEMM needs three dimensions (M, N, K). The exact six-argument
 *   symbol requested in the specification is preserved as a square-RHS
 *   convenience wrapper where K == N. Use execute_sparse_gemm_24_neon_general
 *   for rectangular K.
 *
 * Hot-path contract:
 *   - No data-dependent zero tests are performed.
 *   - Pointers must be valid for the implied shapes and must not alias.
 *   - K must be a multiple of 4 for the general entry point.
 *   - For peak NEON throughput: 64-byte aligned buffers and N % 8 == 0.
 */
#ifndef SPARSE24_H
#define SPARSE24_H

#include <stdint.h>

#if defined(__cplusplus)
  #define SPARSE24_RESTRICT __restrict
#else
  #define SPARSE24_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Requested six-argument ABI.
 * Computes C[M, N] = W[M, N] * B[N, N]  (K == N).
 * N must be a multiple of 4.
 */
void execute_sparse_gemm_24_neon(
    const float*    SPARSE24_RESTRICT packed_weights,
    const float*    SPARSE24_RESTRICT inputs,
    const uint8_t*  SPARSE24_RESTRICT indices,
    float*          SPARSE24_RESTRICT outputs,
    int M,
    int N);

/* General rectangular entry point.
 * Computes C[M, N] = W[M, K] * B[K, N].
 * K must be a multiple of 4.
 */
void execute_sparse_gemm_24_neon_general(
    const float*    SPARSE24_RESTRICT packed_weights,
    const float*    SPARSE24_RESTRICT inputs,
    const uint8_t*  SPARSE24_RESTRICT indices,
    float*          SPARSE24_RESTRICT outputs,
    int M,
    int N,
    int K);

#ifdef __cplusplus
}
#endif

#endif /* SPARSE24_H */
