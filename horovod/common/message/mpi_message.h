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

#ifndef HOROVOD_MPI_MESSAGE_H
#define HOROVOD_MPI_MESSAGE_H

#include <string>
#include <vector>

#include "message.h"

namespace horovod {
namespace common {

// An MPIRequest is a message sent from a rank greater than zero to the
// coordinator (rank zero), informing the coordinator of an operation that
// the rank wants to do and the tensor that it wants to apply the operation to.
class MPIRequest : public HorovodRequest {
public:
  // The request rank is necessary to create a consistent ordering of results,
  // for example in the allgather where the order of outputs should be sorted
  // by rank.
  int32_t request_rank() const;
  void set_request_rank(int32_t value);

  RequestType request_type() const;
  void set_request_type(RequestType value);

  DataType tensor_type() const;
  void set_tensor_type(DataType value);

  const std::string& tensor_name() const;
  void set_tensor_name(const std::string& value);

  int32_t root_rank() const;
  void set_root_rank(int32_t value);

  int32_t device() const;
  void set_device(int32_t value);

  const std::vector<int64_t>& tensor_shape() const;
  void set_tensor_shape(const std::vector<int64_t>& value);
  void add_tensor_shape(int64_t value);

  static void ParseFromBytes(MPIRequest& request, const uint8_t* input);
  static void SerializeToString(const MPIRequest& request, std::string& output);

private:
  int32_t request_rank_ = 0;
  RequestType request_type_ = RequestType::ALLREDUCE;
  DataType tensor_type_ = DataType::HOROVOD_UINT8;
  int32_t root_rank_ = 0;
  int32_t device_ = 0;
  std::string tensor_name_;
  std::vector<int64_t> tensor_shape_;
};

class MPIRequestList {
public:
  const std::vector<MPIRequest>& requests() const;
  void set_requests(const std::vector<MPIRequest>& value);
  void add_request(const MPIRequest& value);
  void emplace_request(MPIRequest&& value);
  bool shutdown() const;
  void set_shutdown(bool value);

  static void ParseFromBytes(MPIRequestList& request_list,
                             const uint8_t* input);
  static void SerializeToString(const MPIRequestList& request_list,
                                std::string& output);

private:
  std::vector<MPIRequest> requests_;
  bool shutdown_ = false;
};

// An MPIResponse is a message sent from the coordinator (rank zero) to a rank
// greater than zero, informing the rank of an operation should be performed
// now. If the operation requested would result in an error (for example, due
// to a type or shape mismatch), then the MPIResponse can contain an error and
// an error message instead.
class MPIResponse : public HorovodResponse {
public:
  ResponseType response_type() const override;
  void set_response_type(ResponseType value);

  // Empty if the type is DONE or SHUTDOWN.
  const std::vector<std::string>& tensor_names() const;
  const std::string tensor_names_string() const;
  void set_tensor_names(const std::vector<std::string>& value);
  void add_tensor_name(const std::string& value);

  // Empty unless response_type is ERROR.
  const std::string& error_message() const override;
  void set_error_message(const std::string& value);

  const std::vector<int32_t>& devices() const override;
  void set_devices(const std::vector<int32_t>& value);
  void add_device(int32_t value);

  // Empty unless response_type is ALLGATHER.
  // These tensor sizes are the dimension zero sizes of all the input matrices,
  // indexed by the rank.
  const std::vector<int64_t>& tensor_sizes() const override;
  void set_tensor_sizes(const std::vector<int64_t>& value);
  void add_tensor_size(int64_t value);

  // To fuse multiple allgather responses
  void add_allgather_response(const MPIResponse& response);

  static void ParseFromBytes(MPIResponse& response, const uint8_t* input);
  static void SerializeToString(const MPIResponse& response,
                                std::string& output);

private:
  ResponseType response_type_ = ResponseType::ALLREDUCE;
  std::vector<std::string> tensor_names_;
  std::string error_message_;
  std::vector<int32_t> devices_;
  std::vector<int64_t> tensor_sizes_;
};

class MPIResponseList {
public:
  const std::vector<MPIResponse>& responses() const;
  void set_responses(const std::vector<MPIResponse>& value);
  void add_response(const MPIResponse& value);
  void emplace_response(MPIResponse&& value);
  bool shutdown() const;
  void set_shutdown(bool value);

  static void ParseFromBytes(MPIResponseList& response_list,
                             const uint8_t* input);
  static void SerializeToString(const MPIResponseList& response_list,
                                std::string& output);

private:
  std::vector<MPIResponse> responses_;
  bool shutdown_ = false;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_MESSAGE_H
