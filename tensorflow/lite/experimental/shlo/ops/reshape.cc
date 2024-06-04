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
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {
absl::Status CheckParameters(const Tensor& operand, Tensor& result) {
        //  for non quantized tensors

  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("reshape"), operand, result));

  if (operand.element_type() != result.element_type()) {
    return absl::FailedPreconditionError(
        "The element type of operand and result must be the same.");
  }
  //         if (!operand.IsPerAxisQuantized()) {

  // }

//   if (operand.quantized_per_axis_element_type().QuantizedDimension() !=
//       result.quantized_per_axis_element_type().QuantizedDimension()) {
//     if (operand.element_type() != result.element_type()) {
//       return absl::FailedPreconditionError(
//           "The element type of operand and result must be the same although "
//           "operand precision and result precision is not same.");
//     }
//   }

  // constraint 2
  // if (operand.SizeInBytes() != result.SizeInBytes()) {
  //   return absl::FailedPreconditionError(
  //       "The element type of operand and result must be of same size.");
  // }
}
  // constraint 3
  // if (operand.IsPerAxisQuantized()) {
  //         const DimensionSize operand_pre_quant_product = Reduce(
  //             Dims(operand, {0, 1, ..., quantization_dimension(operand) -
  //             1}), 1, {0}, [](const DimensionSize& x, const DimensionSize& y)
  //             { return x * y; });
  //         const DimensionSize result_pre_quant_product = Reduce(
  //             Dims(result, {0, 1, ..., quantization_dimension(result) - 1}),
  //             1, {0}, [](const DimensionSize& x, const DimensionSize& y) {
  //             return x * y; });
  //         if (operand_pre_quant_product != result_pre_quant_product) {
  //             return absl::InternalError(
  //                 "Per-axis quantized reshape: Product of pre-quantization
  //                 dimensions mismatch.");
  //         }

  //         if (dim(operand, quantization_dimension(operand)) !=
  //             dim(result, quantization_dimension(result))) {
  //             // Handle error: Quantization dimension sizes don't match
  //             return absl::InternalError(
  //                 "Per-axis quantized reshape: Quantization dimension sizes
  //                 mismatch.");
  //         }

  //         const DimensionSize operand_post_quant_product = Reduce(
  //             Dims(operand, {quantization_dimension(operand) + 1, ...,
  //             rank(operand) - 1}), 1, {0}, [](const DimensionSize& x, const
  //             DimensionSize& y) { return x * y; });
  //         const DimensionSize result_post_quant_product = Reduce(
  //             Dims(result, {quantization_dimension(result) + 1, ...,
  //             rank(result) - 1}), 1, {0}, [](const DimensionSize& x, const
  //             DimensionSize& y) { return x * y; });
  //         if (operand_post_quant_product != result_post_quant_product) {
  //             // Handle error: post-quantization dimensions don't match
  //             return absl::InternalError(
  //                 "Per-axis quantized reshape: Product of post-quantization
  //                 dimensions mismatch.");
  //         }
  //     }
  //     return absl::OkStatus();

// template <DataType storage_type>
// absl::Status PrepareTensorsQuantized(reshapeOp& op, const Tensor& operand,
//                                  Tensor& output) {
//     using StorageT = StorageType<storage_type>;
//     const DimensionSize operand_size = operand.NumElements();
//     const DimensionSize output_size = output.NumElements();
//     op.operand_dequantized_data =
//         std::vector<std::byte>(operand_size * sizeof(StorageT));
//     const Shape operand_shape = operand.shape();
//     Tensor operand_dequantized{
//         .type = TensorType{.shape = operand_shape, .element_type =
//         storage_type}, .data = op.operand_dequantized_data.data()};
//     op.output_dequantized_data =
//         std::vector<std::byte>(output_size * sizeof(StorageT));
//     const Shape output_dequantized_shape = output.shape();
//     Tensor output_dequantized{
//         .type = TensorType{.shape = output_dequantized_shape,
//                             .element_type = storage_type},
//         .data = op.output_dequantized_data.data()};

//     op.operand_dequantized = std::move(operand_dequantized);
//     op.output_dequantized = std::move(output_dequantized);

//     return absl::OkStatus();
// }
template <DataType storage_type>
absl::Status ReshapeTensor(const Tensor& operand, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t operand_rank = operand.Rank();
  const size_t output_rank = output.Rank();
  absl::InlinedVector<Axis, kMaxNumDimensions> operand_index;
  operand_index.resize(operand_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> output_index;
  output_index.resize(output_rank);
  for (size_t k = 0; k < operand_size; ++k) {
    operand.GetNdIndex(k, operand_index);
    output.GetNdIndex(k, output_index);
    output_buffer[output.FlattenIndex(output_index)] =
        operand.Get<storage_type>(operand_index);
  }
  return absl::OkStatus();
}
// template<DataType storage_type,DataType expressed_type>
// void DequantizeOpQuantizePerTensor(reshapeOp& op, const Tensor& operand,
//                                Tensor& output) {
//     using StorageT = StorageType<storage_type>;
//     using ExpressedT = StorageType<expressed_type>;
//     const StorageT* operand_data = operand.GetDataAs<storage_type>();
//     ExpressedT* operand_dequantized_data =
//         op.operand_dequantized.GetDataAs<expressed_type>();
//     StorageT* output_data = output.GetDataAs<storage_type>();
//     ExpressedT* output_dequantized_data =
//         op.output_dequantized.GetDataAs<expressed_type>();
//     const DimensionSize operand_num_elements = operand.NumElements();
//     const StorageT operand_zero_point =
//         operand.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
//     const ExpressedT operand_scale =
//         operand.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

//     for (DimensionSize i = 0; i < operand_num_elements;
//         ++i, ++operand_data, ++operand_dequantized_data) {
//         *operand_dequantized_data =
//             Dequantize(*operand_data, operand_zero_point, operand_scale);
//     }
//     absl::Status status =
//         Evaluate(op, op.operand_dequantized, op.output_dequantized);
//     const DimensionSize output_num_elements = output.NumElements();
//     const StorageT output_zero_point =
//         output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
//     const ExpressedT output_scale =
//         output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
//     const ExpressedT inv_scale = static_cast<ExpressedT>(1 / output_scale);
//     for (DimensionSize i = 0; i < output_num_elements;
//             ++i, ++output_dequantized_data, ++output_data) {
//         *output_data = Quantize<storage_type, expressed_type>(
//             *output_dequantized_data, output_zero_point, inv_scale);
//     }
// }
reshapeOp Create(reshapeOp::Attributes attributes) {
  return {.attributes = attributes};
}
absl::Status Prepare(reshapeOp& op, const Tensor& operand, Tensor& result) {
  absl::Status status = CheckParameters(operand, result);
  
  if(!status.ok()) {
    return status;
  }

  SHLO_REF_RETURN_ON_ERROR(CheckParameters(operand, result));
  return absl::OkStatus();
}

absl::Status Evaluate(reshapeOp& op, const Tensor& operand, Tensor& result) {
  // if (operand.IsQuantized()) {
  //     if (operand.IsPerTensorQuantized()) {
  //     DISPATCH_QUANTIZED(
  //         DequantizeOpQuantizePerTensor,
  //         operand.quantized_per_tensor_element_type().StorageType(),
  //         operand.quantized_per_tensor_element_type().ExpressedType(), op,
  //         operand, result);
  //     }
  // }

  DISPATCH_BOOL_INT_FLOAT(ReshapeTensor, result.tensor_element_type(), operand,
                          result);
  return absl::FailedPreconditionError(
      "stablehlo.dot_general: Unsupported tensor type.");
}
}  // namespace shlo_ref