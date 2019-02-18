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

#include "message.h"

namespace horovod {
namespace common {

const std::string& DataType_Name(DataType value) {
  switch (value) {
    case HOROVOD_UINT8:
      static const std::string uint8("uint8");
      return uint8;
    case HOROVOD_INT8:
      static const std::string int8("int8");
      return int8;
    case HOROVOD_UINT16:
      static const std::string uint16("uint16");
      return uint16;
    case HOROVOD_INT16:
      static const std::string int16("int16");
      return int16;
    case HOROVOD_INT32:
      static const std::string int32("int32");
      return int32;
    case HOROVOD_INT64:
      static const std::string int64("int64");
      return int64;
    case HOROVOD_FLOAT16:
      static const std::string float16("float16");
      return float16;
    case HOROVOD_FLOAT32:
      static const std::string float32("float32");
      return float32;
    case HOROVOD_FLOAT64:
      static const std::string float64("float64");
      return float64;
    case HOROVOD_BOOL:
      static const std::string bool_("bool");
      return bool_;
    case HOROVOD_BYTE:
      static const std::string byte_("byte");
      return byte_;
    case HOROVOD_NULL:
      static const std::string null_("null");
      return null_;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

const std::string& HorovodRequest::RequestType_Name(RequestType value) {
  switch (value) {
    case RequestType::ALLREDUCE:
      static const std::string allreduce("ALLREDUCE");
      return allreduce;
    case RequestType::ALLGATHER:
      static const std::string allgather("ALLGATHER");
      return allgather;
    case RequestType::BROADCAST:
      static const std::string broadcast("BROADCAST");
      return broadcast;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

const std::string& HorovodResponse::ResponseType_Name(ResponseType value) {
  switch (value) {
    case ResponseType::ALLREDUCE:
      static const std::string allreduce("ALLREDUCE");
      return allreduce;
    case ResponseType::ALLGATHER:
      static const std::string allgather("ALLGATHER");
      return allgather;
    case ResponseType::BROADCAST:
      static const std::string broadcast("BROADCAST");
      return broadcast;
    case ResponseType::ERROR:
      static const std::string error("ERROR");
      return error;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

} // namespace common
} // namespace horovod
