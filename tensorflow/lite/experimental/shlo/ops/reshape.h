#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_RESHAPE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_RESHAPE_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

class ReshapeOp {
 public:
  struct Attributes {};
  
};

ReshapeOp Create(ReshapeOp::Attributes);

absl::Status Prepare(ReshapeOp& op, const Tensor& operand, Tensor& output);

absl::Status Evaluate(ReshapeOp& op, const Tensor& operand, Tensor& output);

}  // namespace shlo_ref

#endif 