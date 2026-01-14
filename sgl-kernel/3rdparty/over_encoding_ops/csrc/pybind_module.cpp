#include <torch/torch.h>
#include <pybind11/pybind11.h>

torch::Tensor custom_empty(at::IntArrayRef sizes, torch::ScalarType dtype, int device_id);
torch::Tensor unified_empty(at::IntArrayRef sizes, torch::ScalarType dtype);
torch::Tensor unified_empty_with_device(at::IntArrayRef sizes, torch::ScalarType dtype, int device_id);
void unified_prefetch_to_gpu(torch::Tensor& tensor, int device_id);
void unified_prefetch_to_cpu(torch::Tensor& tensor);

PYBIND11_MODULE(over_encoding_ops_kernel, m) {
    m.doc() = "Over Encoding Ops - Custom memory allocators for PyTorch tensors";

    m.def("custom_empty", &custom_empty, 
          "Create an empty tensor with cudaMallocHost",
          pybind11::arg("sizes"), 
          pybind11::arg("dtype") = torch::kFloat32,
          pybind11::arg("device_id") = 0);
    
    m.def("unified_empty", &unified_empty, 
          "Create an empty tensor using CUDA Unified Memory",
          pybind11::arg("sizes"), 
          pybind11::arg("dtype") = torch::kFloat32);
    
    m.def("unified_empty_with_device", &unified_empty_with_device,
          "Create an empty tensor using CUDA Unified Memory with device preference",
          pybind11::arg("sizes"),
          pybind11::arg("dtype") = torch::kFloat32,
          pybind11::arg("device_id") = 0);
    
    m.def("unified_prefetch_to_gpu", &unified_prefetch_to_gpu,
          "Prefetch unified memory tensor to GPU",
          pybind11::arg("tensor"),
          pybind11::arg("device_id") = 0);
    
    m.def("unified_prefetch_to_cpu", &unified_prefetch_to_cpu,
          "Prefetch unified memory tensor to CPU",
          pybind11::arg("tensor"));
}
