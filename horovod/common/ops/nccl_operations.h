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

#ifndef HOROVOD_NCCL_OPERATIONS_H
#define HOROVOD_NCCL_OPERATIONS_H

#include <nccl.h>

#include "cuda_operations.h"

namespace horovod {
namespace common {

struct NCCLContext {
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;

  void ErrorCheck(std::string op_name, ncclResult_t nccl_result);
};

class NCCLAllreduce : public CUDAAllreduceAsync {
public:
  NCCLAllreduce(NCCLContext* nccl_context, Channel* cpu_channel,
                CUDAContext* cuda_context, HorovodGlobalState* global_state);

protected:
  void InitComm(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) override;

  void DoAllreduce(std::vector<TensorTableEntry>& entries,
                   const void* fused_input_data, void* buffer_data,
                   int64_t& num_elements, size_t& buffer_len) override;

  virtual const std::vector<int32_t> GetDeviceMap(const std::vector<int32_t>& devices);

  virtual void PopulateCommStrategy(int& nccl_rank, int& nccl_size,
                                    Channel::Communicator& nccl_id_bcast_comm);

  NCCLContext* nccl_context_;
  ncclComm_t* nccl_comm_;

  Channel* cpu_channel_;
};

class NCCLHierarchicalAllreduce : public NCCLAllreduce {
public:
  NCCLHierarchicalAllreduce(NCCLContext* nccl_context, Channel* cpu_channel,
                            CUDAContext* cuda_context, HorovodGlobalState* global_state);

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const MPIResponse& response) const override;

protected:
  void DoAllreduce(std::vector<TensorTableEntry>& entries,
                   const void* fused_input_data, void* buffer_data,
                   int64_t& num_elements, size_t& buffer_len) override;

private:
  const std::vector<int32_t> GetDeviceMap(const std::vector<int32_t>& devices) override;

  void PopulateCommStrategy(int& nccl_rank, int& nccl_size,
                            Channel::Communicator& nccl_id_bcast_comm) override;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_NCCL_OPERATIONS_H
