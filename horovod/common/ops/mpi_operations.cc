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

#include "mpi_operations.h"

namespace horovod {
namespace common {

void MPIContext::Allreduce(const void* buffer_data, int64_t num_elements,
                           TensorTableEntry& first_entry, const void* sendbuff,
                           Communicator comm) {
  int op = MPI_Allreduce(sendbuff != nullptr ? sendbuff : MPI_IN_PLACE, (void*) buffer_data,
                         (int) num_elements,
                         GetMPIDataType(first_entry.tensor),
                         first_entry.tensor->dtype() == HOROVOD_FLOAT16 ? mpi_float16_sum : MPI_SUM,
                         GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
}

void MPIContext::Allgatherv(const void *sendbuf, int sendcount, DataType sendtype,
                            void *recvbuf, const int recvcounts[],
                            const int displs[], DataType recvtype,
                            Communicator comm) {
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE, sendcount, GetMPIDataType(sendtype),
                          recvbuf, recvcounts, displs, GetMPIDataType(recvtype),
                          GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
  }
}

void MPIContext::Broadcast(const void* buffer_data, int64_t num_elements,
                           DataType dtype, int root_rank,
                           Communicator comm) {
  int op = MPI_Bcast((void*) buffer_data,
                     (int) num_elements,
                     GetMPIDataType(dtype),
                     root_rank,
                     GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
  }
}

void MPIContext::Barrier(Communicator comm) {
  int op = MPI_Barrier(GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }
}

void MPIContext::AllocateSharedBuffer(int64_t window_size, int element_size, void* baseptr, Communicator comm) {
  MPI_Win_allocate_shared(
      window_size, element_size, MPI_INFO_NULL, GetMPICommunicator(comm),
      baseptr, &window);
}

void MPIContext::FreeSharedBuffer() {
  MPI_Win_fence(0, window);
  MPI_Win_free(&window);
}

void MPIContext::QuerySharedBuffer(int rank, void* baseptr) {
  int disp_unit;
  MPI_Aint winsize;
  MPI_Win_shared_query(window, rank, &winsize, &disp_unit, baseptr);
}

void MPIContext::GetTypeSize(DataType dtype, int* out) {
  MPI_Type_size(GetMPIDataType(dtype), out);
}

MPI_Datatype MPIContext::GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  return GetMPIDataType(tensor->dtype());
}

MPI_Datatype MPIContext::GetMPIDataType(const DataType dtype) {
  switch (dtype) {
    case HOROVOD_UINT8:
      return MPI_UINT8_T;
    case HOROVOD_INT8:
      return MPI_INT8_T;
    case HOROVOD_UINT16:
      return MPI_UINT16_T;
    case HOROVOD_INT16:
      return MPI_INT16_T;
    case HOROVOD_INT32:
      return MPI_INT32_T;
    case HOROVOD_INT64:
      return MPI_INT64_T;
    case HOROVOD_FLOAT16:
      return mpi_float16_t;
    case HOROVOD_FLOAT32:
      return MPI_FLOAT;
    case HOROVOD_FLOAT64:
      return MPI_DOUBLE;
    case HOROVOD_BOOL:
      return MPI_C_BOOL;
    case HOROVOD_BYTE:
      return MPI_BYTE;
    case HOROVOD_NULL:
      return MPI_DATATYPE_NULL;
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                             " is not supported in MPI mode.");
  }
}

MPI_Comm MPIContext::GetMPICommunicator(Communicator comm) {
  switch (comm) {
    case GLOBAL:
      return mpi_comm;
    case LOCAL:
      return local_comm;
    case CROSS:
      return cross_comm;
    default:
      throw std::logic_error("Communicator " + CommunicatorName(comm) +
                             " is not supported in MPI mode.");
  }
}

void DoMPIAllreduce(MPIContext* mpi_context,
                   std::vector<TensorTableEntry>& entries,
                   void* buffer_data, int64_t& num_elements, size_t& buffer_len) {
  auto& first_entry = entries[0];
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? nullptr : first_entry.tensor->data();
  mpi_context->Allreduce(buffer_data, num_elements, first_entry, sendbuf, CommunicationContext::Communicator::GLOBAL);
}

MPIAllreduce::MPIAllreduce(MPIContext* mpi_context,
                           CommunicationContext* comm_context,
                           HorovodGlobalState* global_state)
                           : AllreduceOp(comm_context, global_state), mpi_context_(mpi_context) {}

bool MPIAllreduce::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const MPIResponse& response) const {
  return true;
}

void MPIAllreduce::DoAllreduce(std::vector<TensorTableEntry>& entries,
                               const void* fused_input_data, void* buffer_data,
                               int64_t& num_elements, size_t& buffer_len) {
  RecordEventStart(MPI_ALLREDUCE, entries);
  DoMPIAllreduce(mpi_context_, entries, buffer_data, num_elements, buffer_len);
  RecordEventEnd(MPI_ALLREDUCE, entries);
}

#if HAVE_CUDA
MPI_CUDAAllreduce::MPI_CUDAAllreduce(MPIContext* mpi_context,
                                     CUDAContext* cuda_context,
                                     CommunicationContext* comm_context,
                                     HorovodGlobalState* global_state)
                                     : CUDAAllreduce(cuda_context, comm_context, global_state),
                                       mpi_context_(mpi_context) {}

void MPI_CUDAAllreduce::DoAllreduce(std::vector<TensorTableEntry>& entries,
                                    const void* fused_input_data, void* buffer_data,
                                    int64_t& num_elements, size_t& buffer_len) {
  RecordEventStart(MPI_ALLREDUCE, entries);
  DoMPIAllreduce(mpi_context_, entries, buffer_data, num_elements, buffer_len);
  RecordEventEnd(MPI_ALLREDUCE, entries);
}
#endif

MPIAllgather::MPIAllgather(MPIContext* mpi_context,
                           CommunicationContext* comm_context,
                           HorovodGlobalState* global_state)
                           : AllgatherOp(comm_context, global_state), mpi_context_(mpi_context) {}

bool MPIAllgather::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const MPIResponse& response) const {
  return true;
}

void MPIAllgather::DoAllgatherv(std::vector<TensorTableEntry>& entries,
                                const void *sendbuf, int sendcount, DataType sendtype,
                                void *recvbuf, const int recvcounts[],
                                const int displs[], DataType recvtype) {
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLGATHER);
  mpi_context_->Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                           CommunicationContext::Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);
}

MPIHierarchicalAllgather::MPIHierarchicalAllgather(MPIContext* mpi_context,
                                                   CommunicationContext* comm_context,
                                                   HorovodGlobalState* global_state)
                                                   : HierarchicalAllgather(comm_context, global_state),
                                                     mpi_context_(mpi_context) {}

bool MPIHierarchicalAllgather::Enabled(ParameterManager& param_manager,
                                       std::vector<TensorTableEntry>& entries,
                                       const MPIResponse& response) const {
  return param_manager.HierarchicalAllgather();
}

void MPIHierarchicalAllgather::DoAllgatherv(std::vector<TensorTableEntry>& entries,
                                            const void *sendbuf, int sendcount, DataType sendtype,
                                            void *recvbuf, const int recvcounts[],
                                            const int displs[], DataType recvtype) {
  // Perform the cross-node allgather. If the cluster is homogeneous all
  // local ranks participate, otherwise local rank 0 handles all data
  global_state_->timeline.ActivityStartAll(entries, MPI_CROSS_ALLGATHER);
  if (global_state_->is_homogeneous || global_state_->local_rank == 0) {
    comm_context_->Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                              CommunicationContext::Communicator::CROSS);
  }
  comm_context_->Barrier(CommunicationContext::Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);
}

MPIBroadcast::MPIBroadcast(MPIContext* mpi_context,
                           CommunicationContext* comm_context,
                           HorovodGlobalState* global_state)
                           : BroadcastOp(comm_context, global_state), mpi_context_(mpi_context) {}

bool MPIBroadcast::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const MPIResponse& response) const {
  return true;
}

void MPIBroadcast::DoBroadcast(std::vector<TensorTableEntry>& entries,
                               const void* buffer_data, int64_t num_elements,
                               DataType dtype, int root_rank) {
  global_state_->timeline.ActivityStartAll(entries, MPI_BCAST);
  comm_context_->Broadcast(buffer_data, num_elements, dtype, root_rank,
                           CommunicationContext::Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);
}

} // namespace common
} // namespace horovod
