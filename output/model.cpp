#include "nanocompile_model.hpp"
#include "weights.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace nanocompile {
namespace {
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
}

void inference(const float* input, float* output) {
    alignas(64) static thread_local float scratch[16];
    (void)scratch;

    for (int i_dense1_matmul_add_relu_fused = 0; i_dense1_matmul_add_relu_fused < 1; ++i_dense1_matmul_add_relu_fused) {
        for (int j_dense1_matmul_add_relu_fused = 0; j_dense1_matmul_add_relu_fused < 16; ++j_dense1_matmul_add_relu_fused) {
            float acc = nanocompile_weights::W_B1[j_dense1_matmul_add_relu_fused];
            for (int k_dense1_matmul_add_relu_fused = 0; k_dense1_matmul_add_relu_fused < 8; ++k_dense1_matmul_add_relu_fused) {
                acc += input[i_dense1_matmul_add_relu_fused * 8 + k_dense1_matmul_add_relu_fused] * nanocompile_weights::W_W1[k_dense1_matmul_add_relu_fused * 16 + j_dense1_matmul_add_relu_fused];
            }
            (scratch + 0)[i_dense1_matmul_add_relu_fused * 16 + j_dense1_matmul_add_relu_fused] = relu(acc);
        }
    }

    for (int i_dense2_matmul_add_fused = 0; i_dense2_matmul_add_fused < 1; ++i_dense2_matmul_add_fused) {
        for (int j_dense2_matmul_add_fused = 0; j_dense2_matmul_add_fused < 4; ++j_dense2_matmul_add_fused) {
            float acc = nanocompile_weights::W_B2[j_dense2_matmul_add_fused];
            for (int k_dense2_matmul_add_fused = 0; k_dense2_matmul_add_fused < 16; ++k_dense2_matmul_add_fused) {
                acc += (scratch + 0)[i_dense2_matmul_add_fused * 16 + k_dense2_matmul_add_fused] * nanocompile_weights::W_W2[k_dense2_matmul_add_fused * 4 + j_dense2_matmul_add_fused];
            }
            output[i_dense2_matmul_add_fused * 4 + j_dense2_matmul_add_fused] = acc;
        }
    }

}
}
