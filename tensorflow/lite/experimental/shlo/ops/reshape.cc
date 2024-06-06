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
absl::Status CheckParameters(const Tensor& operand, Tensor& result) {
  //  for non quantized tensors only
  // SHLO_REF_RETURN_ON_ERROR(
  //     CheckSameBaselineType(CheckCtx("reshape"), operand, result));

  if (operand.element_type() != result.element_type()) {
    return absl::FailedPreconditionError(
        "The element type of operand and result must be the same.");
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status ReshapeTensor(const Tensor& operand, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  // const DimensionSize output_size = output.NumElements();

  // const size_t operand_rank = operand.Rank();
  // const size_t output_rank = output.Rank();

  // absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_index;
  // operand_index.resize(operand_rank);

  // absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index;
  // output_index.resize(output_rank);

  for (size_t k = 0; k < operand_size; ++k) {
    // operand.GetNdIndex(k, operand_index);
    // output.GetNdIndex(k, output_index);
    // output_buffer[output.FlattenIndex(output_index)] = operand.GetDataAs<storage_type>(operand_index);
    output_buffer[k] = operand.GetDataAs<storage_type>()[k];
  }

  return absl::OkStatus();
}

ReshapeOp Create(ReshapeOp::Attributes) {
  return {};
}

absl::Status Prepare(ReshapeOp& op, const Tensor& operand, Tensor& result) {
  absl::Status status = CheckParameters(operand, result);

  if (!status.ok()) {
    return status;
  }

  // if (operand.element_type() != result.element_type()) {
  //   return absl::FailedPreconditionError(
  //       "The element type of operand and result must be the same.");
  // }

  SHLO_REF_RETURN_ON_ERROR(CheckParameters(operand, result));
  return absl::OkStatus();
}

absl::Status Evaluate(ReshapeOp& op, const Tensor& operand, Tensor& result) {
  // if (operand.IsQuantized()) {
  //   if (operand.IsPerTensorQuantized()) {
  //     DISPATCH_QUANTIZED(
  //         detail::DequantizeOpQuantizePerTensor,
  //         operand.quantized_per_tensor_element_type().StorageType(),
  //         operand.quantized_per_tensor_element_type().ExpressedType(), op,
  //         operand, result);
  //   } else {
  //     DISPATCH_QUANTIZED(
  //         detail::DequantizeOpQuantizePerAxis,
  //         operand.quantized_per_tensor_element_type().StorageType(),
  //         operand.quantized_per_tensor_element_type().ExpressedType(), op,
  //         operand, result);
  //   }
  // }
  DISPATCH_BOOL_INT_FLOAT(ReshapeTensor, result.StorageType(), operand,
                          result);
  return absl::FailedPreconditionError(
      "stablehlo.reshape: Unsupported tensor type.");
}
}  // namespace shlo_ref