#include "tensorflow/lite/experimental/shlo/ops/reshape.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {
absl::Status CheckParameters(const Tensor& operand, Tensor& output) {
  //  for non quantized tensors only
  // SHLO_REF_RETURN_ON_ERROR(
  //     CheckSameBaselineType(CheckCtx("reshape"), operand, output));

  if (operand.element_type() != output.element_type()) {
    return absl::FailedPreconditionError(
        "The element type of operand and output must be the same.");
  }
  return absl::OkStatus();
}


absl::Status ReshapeTensor(const Tensor& operand, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();

  for (size_t k = 0; k < operand_size; ++k) {
    // operand.GetNdIndex(k, operand_index);
    // output.GetNdIndex(k, output_index);
    // output_buffer[output.FlattenIndex(output_index)] = operand.Get<storage_type>(operand_index);
    
    output_buffer[k] = operand.GetDataAs<storage_type>()[k];
  }

  return absl::OkStatus();
}



ReshapeOp Create(ReshapeOp::Attributes) {
  return {};
}



absl::Status Prepare(ReshapeOp& op, const Tensor& operand, Tensor& output) {
  absl::Status status = CheckParameters(operand, output);

  if (!status.ok()) {
    return status;
  }

  // if (operand.element_type() != output.element_type()) {
  //   return absl::FailedPreconditionError(
  //       "The element type of operand and output must be the same.");
  // }

  SHLO_REF_RETURN_ON_ERROR(CheckParameters(operand, output));
  return absl::OkStatus();
}

absl::Status Evaluate(ReshapeOp& op, const Tensor& operand, Tensor& output) {
  // if (operand.IsQuantized()) {
  //   if (operand.IsPerTensorQuantized()) {
  //     DISPATCH_QUANTIZED(
  //         detail::DequantizeOpQuantizePerTensor,
  //         operand.quantized_per_tensor_element_type().StorageType(),
  //         operand.quantized_per_tensor_element_type().ExpressedType(), op,
  //         operand, output);
  //   } else {
  //     DISPATCH_QUANTIZED(
  //         detail::DequantizeOpQuantizePerAxis,
  //         operand.quantized_per_tensor_element_type().StorageType(),
  //         operand.quantized_per_tensor_element_type().ExpressedType(), op,
  //         operand, output);
  //   }
  // }
  DISPATCH_BOOL_INT_FLOAT(ReshapeTensor, output.StorageType(), operand,
                          output);
  return absl::FailedPreconditionError(
      "stablehlo.reshape: Unsupported tensor type.");
}
}  // namespace shlo_ref