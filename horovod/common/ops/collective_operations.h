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

#ifndef HOROVOD_COLLECTIVE_OPERATIONS_H
#define HOROVOD_COLLECTIVE_OPERATIONS_H

#include <iostream>

#include "../common.h"
#include "../communication_channel.h"
#include "../global_state.h"
#include "../parameter_manager.h"

namespace horovod {
namespace common {

class HorovodOp {
public:
  HorovodOp(HorovodGlobalState* global_state);

  virtual Status Execute(std::vector<TensorTableEntry>& entries, const MPIResponse& response) = 0;

protected:
  HorovodGlobalState* global_state_;
};

class AllreduceOp : public HorovodOp {
public:
  AllreduceOp(HorovodGlobalState* global_state);

  virtual ~AllreduceOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries, const MPIResponse& response);

  virtual bool Enabled(ParameterManager& param_manager,
                       std::vector<TensorTableEntry>& entries,
                       const MPIResponse& response) const = 0;

protected:
  virtual void DoAllreduce(std::vector<TensorTableEntry>& entries,
                           const void* fused_input_data, void* buffer_data,
                           int64_t& num_elements, size_t& buffer_len) = 0;

  virtual void Initialize(std::vector<TensorTableEntry>& entries, const MPIResponse& response);

  virtual Status Finalize(std::vector<TensorTableEntry>& entries);

  virtual void StartMemcpyInFusionBuffer(std::vector<TensorTableEntry>& entries);

  virtual void MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                    std::vector<TensorTableEntry>& entries);

  virtual void EndMemcpyInFusionBuffer(std::vector<TensorTableEntry>& entries);

  virtual void StartMemcpyOutFusionBuffer(std::vector<TensorTableEntry>& entries);

  virtual void MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                     std::vector<TensorTableEntry>& entries);

  virtual void EndMemcpyOutFusionBuffer(std::vector<TensorTableEntry>& entries);
};

class AllgatherOp : public HorovodOp {
public:
  AllgatherOp(HorovodGlobalState* global_state);

  virtual ~AllgatherOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries, const MPIResponse& response);

  virtual bool Enabled(ParameterManager& param_manager,
                       std::vector<TensorTableEntry>& entries,
                       const MPIResponse& response) const = 0;

protected:
  virtual void DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
                           int64_t** entry_component_offsets, int64_t** entry_component_sizes,
                           int64_t total_size, int element_size);

  virtual void DoAllgatherv(std::vector<TensorTableEntry>& entries,
                            const void* sendbuf, int sendcount, DataType sendtype,
                            void* recvbuf, const int recvcounts[],
                            const int displs[], DataType recvtype) = 0;

  virtual int GetElementSize(DataType dtype) const = 0;
};

class BroadcastOp : public HorovodOp {
public:
  BroadcastOp(HorovodGlobalState* global_state);

  virtual ~BroadcastOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries, const MPIResponse& response);

  virtual bool Enabled(ParameterManager& param_manager,
                       std::vector<TensorTableEntry>& entries,
                       const MPIResponse& response) const;

protected:
  virtual void DoBroadcast(std::vector<TensorTableEntry>& entries,
                           const void* buffer_data, int64_t num_elements,
                           DataType dtype, int root_rank) = 0;
};

class ErrorOp : public HorovodOp {
public:
  ErrorOp(HorovodGlobalState* global_state);

  virtual ~ErrorOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries, const MPIResponse& response);
};

class HierarchicalAllgather : public AllgatherOp {
public:
  HierarchicalAllgather(HorovodGlobalState* global_state);

protected:
  void DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
                   int64_t** entry_component_offsets, int64_t** entry_component_sizes,
                   int64_t total_size, int element_size) override;

  virtual void Barrier() = 0;

  virtual void FreeSharedBuffer() = 0;

  virtual void AllocateSharedBuffer(int64_t total_size_in_bytes, int element_size) = 0;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_COLLECTIVE_OPERATIONS_H
