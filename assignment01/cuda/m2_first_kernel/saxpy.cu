#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,    \
                         __LINE__, cudaGetErrorString(error));                \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

__global__ void saxpy(const float *x, float *y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) y[index] = 2.0f * x[index] + y[index];
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <n>\n", argv[0]);
        return EXIT_FAILURE;
    }

    errno = 0;
    char *end = nullptr;
    long parsed_n = std::strtol(argv[1], &end, 10);
    if (errno != 0 || *argv[1] == '\0' || *end != '\0' || parsed_n < 0 ||
        parsed_n > INT_MAX) {
        std::fprintf(stderr, "n must be an integer in [0, %d]\n", INT_MAX);
        return EXIT_FAILURE;
    }
    int n = static_cast<int>(parsed_n);

    // A zero-sized launch is invalid, so it is handled before any allocation.
    if (n == 0) {
        std::printf("SUM=0\n");
        return EXIT_SUCCESS;
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *h_x = static_cast<float *>(std::malloc(bytes));
    float *h_y = static_cast<float *>(std::malloc(bytes));
    if (h_x == nullptr || h_y == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(h_x);
        std::free(h_y);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>((i % 2048) - 1024) * 0.5f;
        h_y[i] = static_cast<float>((i % 1024) - 512);
    }

    float *d_x = nullptr;
    float *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    constexpr int threads_per_block = 256;
    int blocks = (n - 1) / threads_per_block + 1;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    saxpy<<<blocks, threads_per_block>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(h_y[i]);
    std::printf("n=%d kernel_ms=%.3f SUM=%.0f\n", n, kernel_ms, sum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    std::free(h_x);
    std::free(h_y);
    return EXIT_SUCCESS;
}
