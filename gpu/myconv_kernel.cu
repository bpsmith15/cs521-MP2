#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8   
#define TILE_W 8   
#define TILE_C 16  

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C_out x out_h x out_w
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    extern __shared__ float shmem[];
    
    // TO DO : Tiled convolution
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // pixel we are computing
    int out_col = blockIdx.x * TILE_W + tx;
    int out_row = blockIdx.y * TILE_H + ty;
    int n  = blockIdx.z / C_out;                // modified grid size to improve memory access
    int out_channel = blockIdx.z % C_out;       // modified grid size to improve memory access

    // computing out pixel requires area around out pixel (halo)
    int in_tile_height = (TILE_H - 1) * stride + KH;
    int in_tile_width = (TILE_W - 1) * stride + KW; 
    int in_size = in_tile_height * in_tile_width;
    int kernel_size = KH * KW;

    // split shared memory between input and weights
    float* shared_in = shmem;
    float* shared_weights  = shmem + TILE_C * in_size;

    int top = blockIdx.y * TILE_H * stride - pad;
    int left = blockIdx.x * TILE_W * stride - pad;

    int tid = ty * TILE_W + tx;
    int nthreads = TILE_H * TILE_W;
    float acc = 0.0f;

    for (int c0 = 0; c0 < C_in; c0 += TILE_C) {
        int cc = min(TILE_C, C_in - c0);  // channels in this tile

        // Load input tile for cc channels; out-of-bounds -> zero padding
        for (int i = tid; i < cc * in_size; i += nthreads) {
            int channel = i / in_size;
            int rem = i % in_size;
            int in_row = top + rem / in_tile_width;
            int in_col = left + rem % in_tile_width;

            // check in image otherwise 0
            shared_in[i] = (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) ? x[((n * C_in + c0 + channel) * H + in_row) * W + in_col] : 0.0f;
        }

        // Load this output channel's weights for the same cc input channels
        for (int i = tid; i < cc * kernel_size; i += nthreads) {
            int channel = i / kernel_size;
            int k = i % kernel_size;
            shared_weights[i] = w[(out_channel * C_in + c0 + channel) * kernel_size + k];
        }

        // convolve only after shared memory is fully loaded
        __syncthreads();

        // definition of convolution
        if (out_row < out_h && out_col < out_w) {
            for (int c = 0; c < cc; ++c)
                for (int kh = 0; kh < KH; ++kh)
                    for (int kw = 0; kw < KW; ++kw)
                        acc += shared_in[c * in_size + (ty * stride + kh) * in_tile_width + (tx * stride + kw)] * shared_weights[c * kernel_size + kh * KW + kw];
        }

        // wait until convolution is done to move on, don't overwrite shared memory
        __syncthreads(); 
    }

    // check in image before writing
    if (out_row < out_h && out_col < out_w)
        out[((n * C_out + out_channel) * out_h + out_row) * out_w + out_col] = acc;
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    int in_tile_h = (TILE_H - 1) * stride + KH;
    int in_tile_w = (TILE_W - 1) * stride + KW;
    size_t shmem_bytes = (size_t)(TILE_C * in_tile_h * in_tile_w + TILE_C * KH * KW) * sizeof(float);
    TORCH_CHECK(shmem_bytes <= 48 * 1024, "shared memory tile too large: ", shmem_bytes, " bytes");

    dim3 block(8, 8);
    dim3 grid((out_w + block.x - 1)/block.x,
              (out_h + block.y - 1)/block.y,
              N * C_out);

    gemm_gpu_o4_kernel<<<grid, block, shmem_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
