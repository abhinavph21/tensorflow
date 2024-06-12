#include "tensorflow/lite/experimental/shlo/ops/bitcast_convert.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {

void bitcastConvertManyToOne(Tensor& operand, StorageT* output_buffer, absl::InlinedVector<> operand_indices, DataType output_storage_type) {

  auto resultNumBits = SizeOf(output_storage_type) * 8;
  auto operandNumBits = SizeOf(operand.element_type()) * 8;

  // if (resultNumBits % operandNumBits != 0)
  //   report_fatal_error(invalidArgument(
  //       "Unsupported bitcast conversion from %s to %s",
  //       debugString(elements[0].getType()).c_str(), debugString(type).c_str()));


  // APInt resultBits(resultNumBits, 0);
  VariantType result = CreateVariable(output_storage_type);
  std::visit([](auto& var) {
            var = 0;
        }
    }, result);
  
  using vector1 = absl::InlinedVector;

  for (vector1::reverse_iterator it=operand_indices.rbegin(), it!=rend(); it++) {
    
    std::visit([](auto& var) {
        using T = std::decay_t<decltype(var)>;
        // if constexpr (std::is_same_v<T, int32_t>) {
        result |= (static_cast<T>(*it) << operandNumBits);
    }, it);

  }

  // return Element::fromBits(type, resultBits);
}


// void computeAndStoreOutputResult(Tensor& operand, StorageT* output_buffer, absl::InlinedVector<> operand_indices, DataType output_storage_type){

// }


template <DataType output_storage_type>
absl::Status BitcastConvertImplement(Tensor& operand, Tensor& output) {
  // for non-quantized input
  StorageT* output_buffer = output.GetDataAs<output_storage_type>();

  auto resultNumBytes = SizeOf(result.element_type());
  auto operandNumBytes = SizeOf(operand.element_type());

  const DimensionSize output_size = output.NumElements();

//   if (resultNumBits < operandNumBits) {
//     auto resultIt = result.index_begin();
//     for (auto operandIt = operand.index_begin();
//          operandIt != operand.index_end(); ++operandIt) {
//       auto resultElements =
//           bitcastConvertOneToMany(resultElementType, operand.get(*operandIt));
//       for (const auto& resultElement : resultElements)
//         result.set(*resultIt++, resultElement);
//     }
//     return result;
//   }

  if (resultNumBytes > operandNumBytes) {

    auto operandIt = operand.index_begin();

    for (size_t k = 0; k < output_size; ++k) {

      absl::InlinedVector<> operand_indices;

      for (auto i = 0; i < resultNumBits / operandNumBits; ++i)
        operand_indices.push_back(i);

      output.Set<output_storage_type>(k,
                 bitcastConvertManyToOne(operand, output_buffer, operand_indices, output_storage_type));
    }
  }

//   for (auto it = result.index_begin(); it != result.index_end(); ++it)
//     result.set(*it,
//                bitcastConvertOneToOne(resultElementType, operand.get(*it)));
}

absl::Status Create(BitcastConvertOp::Attribute attributes) {};

absl::Status Prepare(BitcastConvertOp& op, Tensor& operand, Tensor& output) {}

absl::Status Evaluate(BitcastConvertOp& op, Tensor& operand, Tensor& output) {
  DISPATCH_BOOL_INT_FLOAT(BitcastConvertImplement, output.StorageType(), operand,
                          output);
}
}  // namespace shlo_ref
