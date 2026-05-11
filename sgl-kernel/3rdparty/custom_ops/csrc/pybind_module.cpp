#include <torch/torch.h>
#include <pybind11/pybind11.h>

torch::Tensor custom_empty(at::IntArrayRef sizes, torch::ScalarType dtype, int device_id);
void build_ngram_with_tree(torch::Tensor ngram_input_ids, torch::Tensor parent_list, torch::Tensor token_list, torch::Tensor current_parrent_list,
    torch::Tensor buffer, int buffer_size,int gram_n, int topk, int i);
void build_ngram_with_target_verify(torch::Tensor ngram_input_ids, torch::Tensor buffer, torch::Tensor draft_token_ids, torch::Tensor tree_mask, 
    torch::Tensor positions, torch::Tensor seq_lens, int gram_n, int draft_token_num, int buffer_size);
void assign_ngram_input_ids_draft_extend_after_decode(torch::Tensor input_ids_gram_decode, torch::Tensor buffer, torch::Tensor draft_token_ids, 
    torch::Tensor accept_length, int gram_n, int buffer_size, bool update_buffer=false);


PYBIND11_MODULE(prc_custom_ops_kernel, m) {
    m.doc() = "PRC_Custom_OP - Custom memory allocators for PyTorch tensors";

    m.def("custom_empty", &custom_empty, 
          "Create an empty tensor with cudaMallocHost",
          pybind11::arg("sizes"), 
          pybind11::arg("dtype") = torch::kFloat32,
          pybind11::arg("device_id") = 0);
    

    m.def("build_ngram_with_tree", &build_ngram_with_tree,
          "Build ngram with tree",
          pybind11::arg("ngram_input_ids"),
          pybind11::arg("parent_list"),
          pybind11::arg("token_list"),
          pybind11::arg("current_parrent_list"),
          pybind11::arg("buffer"),
          pybind11::arg("buffer_size"),
          pybind11::arg("gram_n"),
          pybind11::arg("topk"),
          pybind11::arg("i"));

    m.def("build_ngram_with_target_verify", &build_ngram_with_target_verify,
          "Build ngram with target verify",
          pybind11::arg("ngram_input_ids"),
          pybind11::arg("buffer"),
          pybind11::arg("draft_token_ids"),
          pybind11::arg("tree_mask"),
          pybind11::arg("positions"),
          pybind11::arg("seq_lens"),
          pybind11::arg("gram_n"),
          pybind11::arg("draft_token_num"),
          pybind11::arg("buffer_size"));

    m.def("assign_ngram_input_ids_draft_extend_after_decode", &assign_ngram_input_ids_draft_extend_after_decode,
          "Assign ngram input ids draft extend after decode",
          pybind11::arg("input_ids_gram_decode"),
          pybind11::arg("buffer"),
          pybind11::arg("draft_token_ids"),
          pybind11::arg("accept_length"),
          pybind11::arg("gram_n"),
          pybind11::arg("buffer_size"),
          pybind11::arg("update_buffer") = false);
}
