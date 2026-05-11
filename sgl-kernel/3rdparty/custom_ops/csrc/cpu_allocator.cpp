#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

torch::Tensor custom_empty(at::IntArrayRef sizes, torch::ScalarType dtype = torch::kFloat32, int device_id = 0) {
    int64_t numel = 1;
    for (auto size : sizes) {
        numel *= size;
    }
    
    size_t element_size = torch::elementSize(dtype);
    size_t total_bytes = numel * element_size;
    
    void* host_ptr;
    CUDA_CHECK(cudaMallocHost(&host_ptr, total_bytes));
    
    auto deleter = [](void* p) { 
        if (p) cudaFreeHost(p); 
    };
    
    return torch::from_blob(host_ptr, sizes, deleter, torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id));
}

