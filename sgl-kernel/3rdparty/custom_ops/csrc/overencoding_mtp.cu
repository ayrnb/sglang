#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
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

__global__ void build_ngram_with_tree_kernel(long* ngram_input_ids, long* parent_list, long* token_list, long* current_parrent_list,
    long* buffer, int topk, int gram_n, int buffer_size, int i, int parent_list_stride, int token_list_stride) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if(tid >= topk){
        return;
    }
    long current_pos = current_parrent_list[bid * topk + tid];
    int gram = gram_n - 1 - i;
    if(gram > 0){
        ngram_input_ids[bid * topk + tid] = buffer[(bid+1) * buffer_size - gram];
        return;
    }
    long parent_token;
    for(int gram_ids=0; gram_ids<gram_n-1; gram_ids++){
        int pre_layer_num_node = topk + topk*topk*(i-1); // exclude root node
        int cur_layer_pos = current_pos - pre_layer_num_node;
        int parent_layer_pos = cur_layer_pos / topk;
        int parent_offset = 1 + topk * (i-1);
        int parent_pos = parent_layer_pos + parent_offset;
        //printf("== %d %d %d %d %d ==", bid, tid, parent_list_stride, parent_pos, parent_list[bid * parent_list_stride + parent_pos]);
        parent_pos = parent_list[bid * parent_list_stride + parent_pos];
        parent_token = token_list[bid * token_list_stride + parent_pos];
        current_pos = parent_pos;
        i--;
    }
    ngram_input_ids[bid * topk + tid] = parent_token;
    return;
}

__global__ void build_target_verify_ngram_kernel(
    int64_t* ngram_input_ids,
    int64_t* buffer,
    int64_t* draft_token_ids,
    bool* tree_mask,
    int64_t* positions,
    int64_t* seq_lens,
    int gram_n,
    int draft_token_num,
    int buffer_size)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if(tid != 0){
        return;
    }
    int seq_id = bid / draft_token_num;
    long seq_len, mask_len, mask_offset;
    mask_offset = 0;
    for(int i=0; i<seq_id; i++){
        mask_len = seq_lens[i] + draft_token_num;
        mask_offset += draft_token_num * mask_len;
    }
    seq_len = seq_lens[seq_id];
    mask_len = seq_len + draft_token_num;
    mask_offset += (bid % draft_token_num) * mask_len;

    int target_gram = gram_n;
    long res;
    for(int i=seq_len + draft_token_num - 1; i>=seq_len; i--){
        if(tree_mask[mask_offset + i]){
            target_gram--;
            if(target_gram == 0){
                res = draft_token_ids[seq_id * draft_token_num + i - seq_len];
                //printf("== %d %d %ld %d %d %ld %ld == ", bid, seq_id, mask_offset, seq_id * draft_token_num, i, seq_len, res);
                break;
            }
        }
    }
    if(target_gram != 0){
        res = buffer[(seq_id+1) * buffer_size - target_gram - 1];
    }
    ngram_input_ids[bid] = res;
    return;
}

__global__ void assign_ngram_input_ids_draft_extend_after_decode_kernel(
    int64_t* input_ids,
    int64_t* buffer,
    int64_t* input_ids_gram,
    int32_t* accept_length,
    int gram_n,
    int buffer_size,
    bool update_buffer
){
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int gram = gram_n - 1;
    int accum_accept_len = 0, curr_accept_len;
    for(int i=0; i<bid; i++){
        accum_accept_len += accept_length[i];
    }
    curr_accept_len = accept_length[bid];
    if(tid < curr_accept_len){
        if(tid >= gram){
            input_ids_gram[accum_accept_len + tid] = int64_t(input_ids[accum_accept_len + tid - gram]);
        }else{
            input_ids_gram[accum_accept_len + tid] = buffer[bid * buffer_size + buffer_size - (gram-tid)];
        }
    }
    if(true){
        return;
    }

    if(tid >= buffer_size){
        return;
    }
    long new_buffer[10];
    int remained_size = buffer_size - curr_accept_len;
    if(tid < remained_size){
        new_buffer[tid] = buffer[bid * buffer_size + buffer_size - remained_size + tid];
    } else{
        new_buffer[tid] = int64_t(input_ids[accum_accept_len + tid - remained_size]);
    }
    buffer[bid * buffer_size + tid] = new_buffer[tid];
    return;
}



void build_ngram_with_tree(torch::Tensor ngram_input_ids, torch::Tensor parent_list, torch::Tensor token_list, torch::Tensor current_parrent_list,
    torch::Tensor buffer, int buffer_size,int gram_n, int topk, int i) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int bs = parent_list.size(0);
    int parent_list_stride = parent_list.stride(0);
    int token_list_stride = token_list.stride(0);

    build_ngram_with_tree_kernel<<<bs, 32, 0, stream>>>(ngram_input_ids.data_ptr<long>(), parent_list.data_ptr<long>(),
                                                        token_list.data_ptr<long>(), current_parrent_list.data_ptr<long>(),
                                                        buffer.data_ptr<long>(), topk, gram_n, buffer_size, i, parent_list_stride, token_list_stride);
    return;
}


void build_ngram_with_target_verify(torch::Tensor ngram_input_ids, torch::Tensor buffer, torch::Tensor draft_token_ids, torch::Tensor tree_mask, 
                               torch::Tensor positions, torch::Tensor seq_lens, int gram_n, int draft_token_num, int buffer_size) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int bs = seq_lens.size(0);
    build_target_verify_ngram_kernel<<<bs*draft_token_num, 32, 0, stream>>>(
        ngram_input_ids.data_ptr<int64_t>(), buffer.data_ptr<int64_t>(), draft_token_ids.data_ptr<int64_t>(),
        tree_mask.data_ptr<bool>(), positions.data_ptr<int64_t>(), seq_lens.data_ptr<int64_t>(),
        gram_n, draft_token_num, buffer_size);
    return;
}


void assign_ngram_input_ids_draft_extend_after_decode(torch::Tensor input_ids, torch::Tensor draft_token_ids, torch::Tensor input_ids_gram, 
                                                    torch::Tensor accept_length, int gram_n, int buffer_size, bool update_buffer=false) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int bs = accept_length.numel();
    TORCH_CHECK(buffer_size < 10, "buffer_size should be less than 10");
    assign_ngram_input_ids_draft_extend_after_decode_kernel<<<bs, 32, 0, stream>>>(
        input_ids.data_ptr<int64_t>(), draft_token_ids.data_ptr<int64_t>(), input_ids_gram.data_ptr<int64_t>(),
        accept_length.data_ptr<int32_t>(), gram_n, buffer_size, update_buffer);
    return;
}