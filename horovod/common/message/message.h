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

#ifndef HOROVOD_MESSAGE_H
#define HOROVOD_MESSAGE_H

#include <iostream>
#include <vector>

namespace horovod {
namespace common {

enum DataType {
  HOROVOD_UINT8 = 0,
  HOROVOD_INT8 = 1,
  HOROVOD_UINT16 = 2,
  HOROVOD_INT16 = 3,
  HOROVOD_INT32 = 4,
  HOROVOD_INT64 = 5,
  HOROVOD_FLOAT16 = 6,
  HOROVOD_FLOAT32 = 7,
  HOROVOD_FLOAT64 = 8,
  HOROVOD_BOOL = 9,
  HOROVOD_BYTE = 10,
  HOROVOD_NULL = 11,
};

const std::string& DataType_Name(DataType value);

class HorovodRequest {
public:
  enum RequestType {
    ALLREDUCE = 0,
    ALLGATHER = 1,
    BROADCAST = 2
  };

  static const std::string& RequestType_Name(RequestType value);
};

class HorovodResponse {
public:
  enum ResponseType {
    ALLREDUCE = 0,
    ALLGATHER = 1,
    BROADCAST = 2,
    ERROR = 3
  };

  virtual ~HorovodResponse()=default;

  virtual ResponseType response_type() const = 0;

  virtual const std::string& error_message() const = 0;

  virtual const std::vector<int32_t>& devices() const = 0;

  virtual const std::vector<int64_t>& tensor_sizes() const = 0;

  static const std::string& ResponseType_Name(ResponseType value);
};


} // namespace common
} // namespace horovod

#endif //HOROVOD_MESSAGE_H
