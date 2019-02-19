// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "nccl_operations.h"

namespace horovod {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
    case HOROVOD_INT32:
      return ncclInt32;
    case HOROVOD_INT64:
      return ncclInt64;
    case HOROVOD_FLOAT16:
      return ncclFloat16;
    case HOROVOD_FLOAT32:
      return ncclFloat32;
    case HOROVOD_FLOAT64:
      return ncclFloat64;
    default:
      throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                             " is not supported in NCCL mode.");
  }
}

void NCCLContext::ErrorCheck(std::string op_name, ncclResult_t nccl_result) {
  if (nccl_result != ncclSuccess) {
    throw std::logic_error(std::string(op_name) + " failed: " + ncclGetErrorString(nccl_result));
  }
}

NCCLAllreduce::NCCLAllreduce(NCCLContext *nccl_context,
                             Channel *cpu_channel,
                             CUDAContext *cuda_context,
                             HorovodGlobalState *global_state)
    : CUDAAllreduceAsync(cuda_context, global_state),
      nccl_context_(nccl_context), cpu_channel_(cpu_channel) {}

void NCCLAllreduce::InitComm(std::vector<TensorTableEntry> &entries, const std::vector<int32_t> &devices) {
  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> nccl_device_map = GetDeviceMap(devices);

  // Ensure NCCL communicator is in the map before executing reduction.
  ncclComm_t &nccl_comm = nccl_context_->nccl_comms[nccl_device_map];
  if (nccl_comm == nullptr) {
    auto &timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_NCCL);

    int nccl_rank, nccl_size;
    Channel::Communicator nccl_id_bcast_comm;
    PopulateCommStrategy(nccl_rank, nccl_size, nccl_id_bcast_comm);

    ncclUniqueId nccl_id;
    if (nccl_rank == 0) {
      nccl_context_->ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&nccl_id));
    }

    cpu_channel_->Broadcast((void *) &nccl_id, sizeof(nccl_id), HOROVOD_BYTE, 0, nccl_id_bcast_comm);

    ncclComm_t new_nccl_comm;
    auto nccl_result = ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank);
    nccl_context_->ErrorCheck("ncclCommInitRank", nccl_result);
    nccl_comm = new_nccl_comm;

    // Barrier helps NCCL to synchronize after initialization and avoid
    // deadlock that we've been seeing without it.
    cpu_channel_->Barrier(Channel::Communicator::GLOBAL);

    timeline.ActivityEndAll(entries);
  }

  nccl_comm_ = &nccl_comm;
}

void NCCLAllreduce::DoAllreduce(std::vector<TensorTableEntry> &entries,
                                const void *fused_input_data, void *buffer_data,
                                int64_t &num_elements, size_t &buffer_len) {
  auto &first_entry = entries[0];
  auto nccl_result = ncclAllReduce(fused_input_data, buffer_data,
                                   (size_t) num_elements,
                                   GetNCCLDataType(first_entry.tensor), ncclSum,
                                   *nccl_comm_, *stream_);
  nccl_context_->ErrorCheck("ncclAllReduce", nccl_result);
  RecordEventEnd(NCCL_ALLREDUCE, entries);
}

const std::vector<int32_t> NCCLAllreduce::GetDeviceMap(const std::vector<int32_t> &devices) {
  return devices;
}

void NCCLAllreduce::PopulateCommStrategy(int &nccl_rank, int &nccl_size,
                                         Channel::Communicator &nccl_id_bcast_comm) {
  nccl_rank = global_state_->rank;
  nccl_size = global_state_->size;
  nccl_id_bcast_comm = Channel::Communicator::GLOBAL;
}

NCCLHierarchicalAllreduce::NCCLHierarchicalAllreduce(NCCLContext *nccl_context, Channel *cpu_channel,
                                                     CUDAContext *cuda_context, HorovodGlobalState *global_state)
    : NCCLAllreduce(nccl_context, cpu_channel,
                    cuda_context, global_state) {}

bool NCCLHierarchicalAllreduce::Enabled(ParameterManager &param_manager,
                                        std::vector<TensorTableEntry> &entries,
                                        const MPIResponse &response) const {
  if (!NCCLAllreduce::Enabled(param_manager, entries, response)) {
    return false;
  }
  return param_manager.HierarchicalAllreduce();
}

void NCCLHierarchicalAllreduce::DoAllreduce(std::vector<TensorTableEntry> &entries,
                                            const void *fused_input_data, void *buffer_data,
                                            int64_t &num_elements, size_t &buffer_len) {
  auto &first_entry = entries[0];
  int element_size;
  cpu_channel_->GetTypeSize(first_entry.tensor->dtype(), &element_size);

  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (global_state_->is_homogeneous && entries.size() > 1) {
    // Making sure the number of elements is divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for improved performance
    int div = global_state_->local_size * FUSION_BUFFER_ATOMIC_UNIT;
    num_elements = ((num_elements + div - 1) / div) * div;
    buffer_len = num_elements * element_size;
  }

  // Split the elements into two groups: num_elements_per_rank*local_size,
  // and num_elements_remaining. Cross-node reduction for the first group
  // is done by all local_rank's in parallel, while for the second group
  // it it is only done by the root_rank. If the cluster is not
  // homogeneous first group is zero, and root_rank is 0.

  // Homogeneous case:
  // For the part of data divisible by local_size, perform NCCL
  // ReduceScatter - Parallelized MPI Allreduce - NCCL Allgather. For the
  // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
  // MPI Allreduce (across rank (local_size-1)'s), and NCCL Bcast

  int64_t num_elements_per_rank =
      global_state_->is_homogeneous
      ? num_elements / global_state_->local_size
      : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void *buffer_data_at_rank_offset =
      (uint8_t *) buffer_data +
      buffer_len_per_rank * global_state_->local_rank;

  int64_t num_elements_remaining =
      global_state_->is_homogeneous
      ? num_elements % global_state_->local_size
      : num_elements;

  size_t buffer_len_remaining = element_size * num_elements_remaining;

  void *buffer_data_remainder =
      (uint8_t *) buffer_data +
      buffer_len_per_rank * global_state_->local_size;

  void *fused_input_data_remainder =
      (uint8_t *) fused_input_data +
      buffer_len_per_rank * global_state_->local_size;

  int root_rank =
      global_state_->is_homogeneous ? global_state_->local_size - 1 : 0;
  bool is_root_rank = global_state_->local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;
  int64_t total_buffer_len =
      is_root_rank ? buffer_len_per_rank + buffer_len_remaining
                   : buffer_len_per_rank;

  auto &timeline = global_state_->timeline;
  if (num_elements_per_rank > 0) {
    auto nccl_result = ncclReduceScatter(fused_input_data,
                                         buffer_data_at_rank_offset,
                                         (size_t) num_elements_per_rank,
                                         GetNCCLDataType(first_entry.tensor),
                                         ncclSum, *nccl_comm_, *stream_);
    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result);
    RecordEventEnd(NCCL_REDUCESCATTER, entries);
  }

  if (num_elements_remaining > 0) {
    // Reduce the remaining data at local_size-1 to append to
    // existing buffer
    auto nccl_result = ncclReduce(fused_input_data_remainder,
                                  buffer_data_remainder,
                                  (size_t) num_elements_remaining,
                                  GetNCCLDataType(first_entry.tensor), ncclSum,
                                  root_rank, *nccl_comm_, *stream_);
    nccl_context_->ErrorCheck("ncclReduce", nccl_result);
    RecordEventEnd(NCCL_REDUCE, entries);
  }

  if (global_state_->is_homogeneous || is_root_rank) {
    // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
    // a buffer is not safe since the tensor can be arbitrarily large.
    host_buffer_ = malloc(total_buffer_len);

    // Synchronize.
    cuda_context_->WaitForEvents(event_queue_, entries, timeline);

    // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
    // api-sync-behavior.html#api-sync-behavior__memcpy-async,
    // cudaMemcpyAsync is synchronous with respect to the host, so we
    // memcpy (effectively) synchronously to generate an accurate timeline
    timeline.ActivityStartAll(entries, MEMCPY_IN_HOST_BUFFER);
    cuda_context_->ErrorCheck("cudaMemcpyAsync",
                              cudaMemcpyAsync(host_buffer_, buffer_data_at_rank_offset,
                                              total_buffer_len, cudaMemcpyDeviceToHost,
                                              *stream_));
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
    cpu_channel_->Allreduce(host_buffer_, total_num_elements, first_entry,
                            nullptr, Channel::Communicator::CROSS);
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MEMCPY_OUT_HOST_BUFFER);
    cuda_context_->ErrorCheck("cudaMemcpyAsync",
                              cudaMemcpyAsync(buffer_data_at_rank_offset, host_buffer_,
                                              total_buffer_len, cudaMemcpyHostToDevice,
                                              *stream_));
    timeline.ActivityEndAll(entries);
  }

  if (num_elements_per_rank > 0) {
    nccl_context_->ErrorCheck("ncclAllGather",
                              ncclAllGather(buffer_data_at_rank_offset, buffer_data,
                                            (size_t) num_elements_per_rank,
                                            GetNCCLDataType(first_entry.tensor),
                                            *nccl_comm_, *stream_));
    RecordEventEnd(NCCL_ALLGATHER, entries);
  }
  if (num_elements_remaining > 0) {
    nccl_context_->ErrorCheck("ncclBcast",
                              ncclBcast(buffer_data_remainder,
                                        (size_t) num_elements_remaining,
                                        GetNCCLDataType(first_entry.tensor), root_rank,
                                        *nccl_comm_, *stream_));
    RecordEventEnd(NCCL_BCAST, entries);
  }
}

const std::vector<int32_t> NCCLHierarchicalAllreduce::GetDeviceMap(const std::vector<int32_t> &devices) {
  std::vector<int32_t> nccl_device_map;
  nccl_device_map.reserve(global_state_->local_comm_ranks.size());
  for (int rank : global_state_->local_comm_ranks) {
    nccl_device_map.push_back(devices[rank]);
  }
  return nccl_device_map;
}

void NCCLHierarchicalAllreduce::PopulateCommStrategy(int &nccl_rank, int &nccl_size,
                                                     Channel::Communicator &nccl_id_bcast_comm) {
  nccl_rank = global_state_->local_rank;
  nccl_size = global_state_->local_size;
  nccl_id_bcast_comm = Channel::Communicator::LOCAL;
}

} // namespace common
} // namespace horovod
