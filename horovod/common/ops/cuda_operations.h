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

#ifndef HOROVOD_CUDA_OPERATIONS_H
#define HOROVOD_CUDA_OPERATIONS_H

#include <queue>
#include <unordered_map>

#include <cuda_runtime.h>

#include "collective_operations.h"

namespace horovod {
namespace common {

struct CUDAContext {
  cudaError_t GetCudaEvent(cudaEvent_t *event);

  cudaError_t ReleaseCudaEvent(cudaEvent_t event);

  // The CUDA stream used for data transfers and within-allreduce operations.
  // A naive implementation would use the TensorFlow StreamExecutor CUDA
  // stream. However, the allreduce and allgather require doing memory copies
  // and kernel executions (for accumulation of values on the GPU). However,
  // the subsequent operations must wait for those operations to complete,
  // otherwise MPI (which uses its own stream internally) will begin the data
  // transfers before the CUDA calls are complete. In order to wait for those
  // CUDA operations, if we were using the TensorFlow stream, we would have to
  // synchronize that stream; however, other TensorFlow threads may be
  // submitting more work to that stream, so synchronizing on it can cause the
  // allreduce to be delayed, waiting for compute totally unrelated to it in
  // other parts of the graph. Overlaying memory transfers and compute during
  // backpropagation is crucial for good performance, so we cannot use the
  // TensorFlow stream, and must use our own stream.
  std::unordered_map<int, cudaStream_t> streams;

  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;

  void ErrorCheck(std::string op_name, cudaError_t cuda_result);

  void RecordEvent(std::queue<std::pair<std::string, cudaEvent_t>> &event_queue, std::string name, cudaStream_t stream);

  void WaitForEvents(std::queue<std::pair<std::string, cudaEvent_t>> &event_queue,
                     std::vector<TensorTableEntry> &entries, Timeline &timeline);
};

class CUDAAllreduce : public AllreduceOp {
public:
  CUDAAllreduce(CUDAContext *context,
                HorovodGlobalState *global_state);

  bool Enabled(ParameterManager &param_manager,
               std::vector<TensorTableEntry> &entries,
               const MPIResponse &response) const override;

protected:
  void Initialize(std::vector<TensorTableEntry> &entries, const MPIResponse &response) override;

  void MemcpyInFusionBuffer(void *buffer_data_at_offset, TensorTableEntry &e,
                            std::vector<TensorTableEntry> &entries) override;

  void MemcpyOutFusionBuffer(void *buffer_data_at_offset, TensorTableEntry &e,
                             std::vector<TensorTableEntry> &entries) override;

  void StreamSynchronize(std::vector<TensorTableEntry> &entries) override;

  void InitCUDA(std::vector<TensorTableEntry> &entries);

  virtual void InitComm(std::vector<TensorTableEntry> &entries, const std::vector<int32_t> &devices) = 0;

  struct CUDAContext *cuda_context_;
};

// Implementation of the Allreduce operation that does not block using StreamSynchronize after each step
// (memcpy into fusion buffer, allreduce, memcpy out of fusion buffer), and instead relies in a separate
// finalize thread to handle synchronization at the end of the operation.
class CUDAAllreduceAsync : public CUDAAllreduce {
public:
  CUDAAllreduceAsync(CUDAContext *context,
                     HorovodGlobalState *global_state);

protected:
  void Initialize(std::vector<TensorTableEntry> &entries, const MPIResponse &response) override;

  Status Finalize(std::vector<TensorTableEntry> &entries) override;

  void StreamSynchronize(std::vector<TensorTableEntry> &entries) override;

  void RecordEventStart(std::string event_name, std::vector<TensorTableEntry> &entries) override;

  void RecordEventEnd(std::string event_name, std::vector<TensorTableEntry> &entries) override;

  std::queue<std::pair<std::string, cudaEvent_t>> event_queue_;
  cudaStream_t *stream_;
  void *host_buffer_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_CUDA_OPERATIONS_H
