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

torch::Tensor unified_empty(at::IntArrayRef sizes, torch::ScalarType dtype = torch::kFloat32) {
    int64_t numel = 1;
    for (auto size : sizes) {
        numel *= size;
    }
    
    size_t element_size = torch::elementSize(dtype);
    size_t total_bytes = numel * element_size;
    
    void* unified_ptr;
    CUDA_CHECK(cudaMallocManaged(&unified_ptr, total_bytes));
    
    // 设置首选位置为CPU，让数据初始放在CPU内存上
    CUDA_CHECK(cudaMemAdvise(unified_ptr, total_bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    // 预取到CPU确保数据在CPU上
    CUDA_CHECK(cudaMemPrefetchAsync(unified_ptr, total_bytes, cudaCpuDeviceId));
    
    auto deleter = [](void* p) { 
        if (p) cudaFree(p); 
    };
    return torch::from_blob(unified_ptr, sizes, deleter, torch::TensorOptions().dtype(dtype));
}

torch::Tensor unified_empty_with_device(at::IntArrayRef sizes, torch::ScalarType dtype = torch::kFloat32, int device_id = 0) {
    int64_t numel = 1;
    for (auto size : sizes) {
        numel *= size;
    }
    
    size_t element_size = torch::elementSize(dtype);
    size_t total_bytes = numel * element_size;
    
    // Set device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    void* unified_ptr;
    CUDA_CHECK(cudaMallocManaged(&unified_ptr, total_bytes));
    cudaMemAdvise(unified_ptr, total_bytes, cudaMemAdviseSetAccessedBy, device_id);
    
    auto deleter = [](void* p) { 
        if (p) cudaFree(p); 
    };
    
    return torch::from_blob(unified_ptr, sizes, deleter, torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id));
}

void unified_prefetch_to_gpu(torch::Tensor& tensor, int device_id = 0) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous for prefetching");
    }
    
    void* data_ptr = tensor.data_ptr();
    size_t size_bytes = tensor.numel() * tensor.element_size();
    
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemPrefetchAsync(data_ptr, size_bytes, device_id));
}

void unified_prefetch_to_cpu(torch::Tensor& tensor) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous for prefetching");
    }
    
    void* data_ptr = tensor.data_ptr();
    size_t size_bytes = tensor.numel() * tensor.element_size();
    
    CUDA_CHECK(cudaMemPrefetchAsync(data_ptr, size_bytes, cudaCpuDeviceId));
}

