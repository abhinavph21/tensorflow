#include "Eigen/Core"  // from @eigen_archive

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

#include <iostream>

namespace tflite {
namespace ops {
namespace builtin {

namespace stablehlo_reshape {

namespace {

constexpr int kOperandTensor = 0;
constexpr int kOutputTensor = 0;

using TfLiteIntArrayUniquePtr =
    std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)>;



TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {

  std:: cout << "entered prepare" << std::endl;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    std:: cout << " prepare after inputs" << std::endl;


  const TfLiteTensor* operand;
  // kOperandTensor = 0
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));

    std:: cout << " prepare got inputs from operand" << std::endl;

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

    std:: cout << " prepare got output" << std::endl;
    

  if (operand->dims->size != output->dims->size) {
    
    return TfLiteStatus::kTfLiteError;
  }
  // ResizeTensor takes ownership of result_shape
  // TF_LITE_ENSURE_STATUS(
  //     context->ResizeTensor(context, output, result_shape.release()));

    std:: cout << " size check done" << std::endl;

  return TfLiteStatus::kTfLiteOk;
}





template <typename IndexType, typename DataType>
TfLiteStatus EvalWithTypes(TfLiteContext* context, TfLiteNode* node) {
  //  2 for param, operand
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  int operand_rank = operand->dims->size;

  RuntimeShape operand_shape = GetTensorShape(operand);

  Index<IndexType> operand_index = Index<IndexType>(operand_rank, 0);

  int result_rank = output->dims->size;

  RuntimeShape result_runtime_shape(result_rank, output->dims->data);

  Index<IndexType> result_index = Index<IndexType>(result_rank, 0);

  do {
    const DataType* operand_data = GetTensorData<DataType>(operand);

    IndexType flat_operand_index = TensorIndexToFlat(
        operand_index.data(), operand_index.size(), GetTensorShape(operand));

    DataType* result_data = GetTensorData<DataType>(output);

    IndexType flat_result_index = TensorIndexToFlat(
        result_index.data(), result_index.size(), GetTensorShape(output));

    result_data[flat_result_index] = operand_data[flat_operand_index];

  } while (
      NextIndex(result_rank, result_runtime_shape.DimsData(),
                result_index.data()) &&
      NextIndex(operand_rank, operand_shape.DimsData(), operand_index.data()));

  return TfLiteStatus::kTfLiteOk;
}



template <typename IndexType>
TfLiteStatus EvalWithIndexType(TfLiteContext* context, TfLiteNode* node,
                               TfLiteType data_type) {
  switch (data_type) {
    
    case kTfLiteFloat16:
      return EvalWithTypes<IndexType, Eigen::half>(context, node);
    case kTfLiteFloat32:
      return EvalWithTypes<IndexType, float>(context, node);
    case kTfLiteFloat64:
      return EvalWithTypes<IndexType, double>(context, node);
    case kTfLiteInt8:
      return EvalWithTypes<IndexType, int8_t>(context, node);
    case kTfLiteInt16:
      return EvalWithTypes<IndexType, int16_t>(context, node);
    case kTfLiteInt32:
      return EvalWithTypes<IndexType, int32_t>(context, node);
    case kTfLiteInt64:
      return EvalWithTypes<IndexType, int64_t>(context, node);
    case kTfLiteUInt8:
      return EvalWithTypes<IndexType, uint8_t>(context, node);
    case kTfLiteUInt16:
      return EvalWithTypes<IndexType, uint16_t>(context, node);
    case kTfLiteUInt32:
      return EvalWithTypes<IndexType, uint32_t>(context, node);
    case kTfLiteUInt64:
      return EvalWithTypes<IndexType, uint64_t>(context, node);
    default:
    //   TF_LITE_KERNEL_LOG(
    //       context, "(Index Type: %s, Data Type: %s) currently not supported.\n",
    //       IndexType, TfLiteTypeGetName(data_type));
      return TfLiteStatus::kTfLiteError;
  }
}



TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {

  const TfLiteTensor* operand;
  
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kOperandTensor, &operand));

  TfLiteType data_type = operand->type;

  return EvalWithIndexType<int32_t>(context, node, data_type);

}

}  // namespace
}  // namespace stablehlo_reshape

TfLiteRegistration* Register_STABLEHLO_RESHAPE() {
  static TfLiteRegistration r = {nullptr, nullptr, stablehlo_reshape::Prepare,
                                 stablehlo_reshape::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
