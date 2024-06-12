#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BITCAST_CONVERT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BITCAST_CONVERT_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref{
    class BitcastConvertOp{
        public:
            struct Attribute {};
    };

    BitcastConvertOp Create(BitcastConvertOp::Attribute);

    absl::Status Prepare(BitcastConvertOp& op, Tensor& operand, Tensor& output);

    absl::Status Evaluate(BitcastConvertOp& op,Tensor& operand, Tensor& output);
}

#endif