#include "absl/status/status.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/reshape.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"


using shlo_ref::testing::StatusIs;
using testing::Eq;
using testing::FloatEq;
using testing::Pointwise;

namespace shlo_ref{
    template <>
struct ParamName<ReshapeOp> {
  static std::string Get() { return "reshape"; }
};
    namespace{
        template<class T>
        struct NonQuantizedIntReshapeTest : ::testing::Test {};
        TYPED_TEST_SUITE(NonQuantizedIntReshapeTest, IntTestTypes, TestParamNames);

        TYPED_TEST(NonQuantizedIntReshapeTest, IntTestTypesTensorrsWork1){
            using StorageT = typename TypeParam::StorageT;
            const Shape shape_operand({2,3,2});
            const Shape shape_r({3,2,2});
            Vector<StorageT> operand_data =
                Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            Vector<StorageT> output_data(shape_r.NumElements());

            Tensor operand{.type = TensorType{.shape = shape_operand,
                                              .element_type = TypeParam::kStorage},
                            .data = operand_data.data()};
            Tensor output_tensor{
                .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
                .data = output_data.data()};

            auto op = Create(ReshapeOp::Attributes{});

            Vector<StorageT> expected_data = Vector<StorageT>{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
            ASSERT_OK(Prepare(op, operand, output_tensor));
            ASSERT_OK(Evaluate(op, operand, output_tensor));
            EXPECT_THAT(output_data, expected_data);

        }
        
        TYPED_TEST(NonQuantizedIntReshapeTest, IntTestTypesRaiseAnError1) {

            using StorageT = typename TypeParam::StorageT;

            const Shape  shape_operand({2,3,2});
            const Shape shape_r({3, 2, 2});
            Vector<StorageT> operand_data = Vector<StorageT>{1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            Vector<StorageT> output_data(shape_r.NumElements());
            Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                            .data = operand_data.data()};
            Tensor output_tensor{
                .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
                .data = output_data.data()};

                auto op = Create(ReshapeOp::Attributes{});
                const absl::Status status = Prepare(op, operand, output_tensor);
                EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                                        absl::StatusCode::kFailedPrecondition));
                EXPECT_THAT(status.message(),
                            "stablehlo.reshape: The output shape should be equal to the "
                            "expected shape.");
        }
        // TYPED_TEST(QuantizedIntReshapeTest, QuantizedTestTypesTensorsWork2) {
        //     using StorageT = typename TypeParam::StorageT;
        //     using ExpressedT = typename TypeParam::ExpressedT;

        //     const Shape shape_operand({1, 3, 2});
        //     const Shape shape_r({3, 1, 2});
        //     Vector<StorageT> operand_data = Vector<StorageT>{1, 2, 3, 4, 5, 6};
        //     Vector<StorageT> output_data(shape_r.NumElements());
        //     std::vector<StorageT> zeroes = {0};
        //     std::vector<float> scales = {1.2f};

        //     QuantizedElementTypePerAxis tensor_type_axis(
        //         TypeParam::kStorage, zeroes, TypeParam::kExpressed, scales, 0);
        //     QuantizedElementTypePerAxis tensor_type_axis_output(
        //         TypeParam::kStorage, zeroes, TypeParam::kExpressed, scales, 1);
            
        //     Tensor operand{
        //         .type = QuantizedPerAxisTensorType{.shape = shape_operand,
        //                                             .element_type = tensor_type_axis},
        //         .data = operand_data.data()};
        //     Tensor output_tensor{
        //         .type = 
        //             QuantizedPerAxisTensorType{.shape = shape_r,
        //                                       .element_type = tensor_type_axis_output},
        //         .data = output_data.data()};
        //     auto op = Create(reshapeOp::Attributes{});
        //     Vector<float> expected_data{1.2002f,  2.40039f, 3.60156f,
        //                                 4.80078f, 6, 7.20312f};
        //     Vector<float> expected_quantized(shape_r.NumElements());
        //     std::transform(expected_data.begin(), expected_data.end(),
        //                     expected_quantized.begin(), [&](float val) {
        //                     return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
        //                         static_cast<ExpressedT>(val), zeroes[0],
        //                         static_cast<ExpressedT>(
        //                             (1.0) / static_cast<ExpressedT>(scales[0])));
        //                     });
        //         ASSERT_OK(Prepare(op, operand, output_tensor));
        //         ASSERT_OK(Evaluate(op, operand, output_tensor));
        //         EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));

        // }


        // TYPED_TEST(NonQuantizedIntReshapeTest, IntTestTypesRaiseAnError2) {
        // using StorageT = typename TypeParam::StorageT;

        // const Shape shape_operand({2, 3, 2});
        // const Shape shape_r({2, 2, 3});
        // Vector<StorageT> operand_data = Vector<StorageT>{1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        // Vector<StorageT> output_data(shape_r.NumElements());

        // Tensor operand{.type = TensorType{.shape = shape_operand,
        //                                     .element_type = TypeParam::kStorage},
        //                 .data = operand_data.data()};
        // Tensor output_tensor{
        //     .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
        //     .data = output_data.data()};

        // auto op = Create(reshapeOp::Attributes{});

        // const absl::Status status = Prepare(op, operand, output_tensor);
        // EXPECT_THAT(status, shlo_ref::testing::StatusIs(
        //                         absl::StatusCode::kFailedPrecondition));
        // EXPECT_THAT(status.message(),
        //             "stablehlo.reshape: The output shape should be different to the "
        //             " operand shape.");
        // }


        // using kBF16TestTypes = ::testing::Types<TestParam<DataType::kBF16>>;
        // template <class T>
        // struct NonQuantizedkBF16ReshapeTest : ::testing::Test {};
        // TYPED_TEST_SUITE(NonQuantizedkBF16ReshapeTest, kBF16TestTypes,
        //                 TestParamNames);
        // TYPED_TEST(NonQuantizedkBF16ReshapeTest, kBF16TestTypesTensorsWork1) {
        //     using StorageT = typename TypeParam::StorageT;
        //     const Shape shape_operand({2, 3});
        //     const Shape shape_r({3, 2});
        //     Vector<float> operand_data_float{9.921870e-01, 8.828120e-01,  -1.179690e+00,
        //                                     1.726560e+00, -1.156250e+00, 7.656250e-01};
        //     Vector<StorageT> operand_data(operand_data_float.begin(),
        //                                     operand_data_float.end());
        //     Vector<StorageT> output_data(shape_r.NumElements());

        //     Tensor operand{.type = TensorType{.shape = shape_operand,
        //                             .element_type = TypeParam::kStorage},
        //                     .data = operand_data.data()};

        //     Tensor output_tensor{
        //         .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
        //         .data = output_data.data()};

        //     auto op = Create(reshapeOp::Attributes{});  
        //     Vector<float> expected_data_float{9.921870e-01,  1.726560e+00,  8.828120e-01,
        //                                     -1.156250e+00, -1.179690e+00, 7.656250e-01};
        //     Vector<StorageT> expected_data(expected_data_float.begin(),
        //                                    expected_data_float.end());
            
        //     ASSERT_OK(Prepare(op, operand, output_tensor));
        //     ASSERT_OK(Evaluate(op, operand, output_tensor));
        //     EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
        // }
        
    //     template <class T>
    //     struct QuantizedIntReshapeTest : ::testing::Test {};
    //     TYPED_TEST_SUITE(QuantizedIntReshapeTest, QuantizedTestTypes, TestParamNames);

    //     TYPED_TEST(QuantizedIntReshapeTest, QuantizedTestTypesTensorsWork1){
    //         using StorageT = typename TypeParam::StorageT;
    //         using ExpressedT = typename TypeParam::ExpressedT;

    //         const Shape shape_operand({2,3,2});
    //         const Shape shape_r({3,2,2});
    //         Vector<StorageT> operand_data =Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    //         Vector<StorageT> output_data(shape_r.NumElements());
    //         const ExpressedT scale = static_cast<ExpressedT>(1.5);
    //         const StorageT zero_point = static_cast<StorageT>(0);
    //         const QuantizedElementTypePerTensor tensor_type =
    //         QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
    //                                        TypeParam::kExpressed, scale);

    //         Tensor operand{
    //             .type = QuantizedPerTensorTensorType{.shape = shape_operand,
    //                                                 .element_type = tensor_type},
    //             .data = operand_data.data()};
    //         Tensor output_tensor{
    //             .type = QuantizedPerTensorTensorType{.shape = shape_r,
    //                                                 .element_type = tensor_type},
    //             .data = output_data.data()};
    //         auto op = Create(reshapeOp::Attributes{});
    //         Vector<StorageT> expected_data{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12 };

    //         ASSERT_OK(Prepare(op, operand, output_tensor));
    //         ASSERT_OK(Evaluate(op, operand, output_tensor));
    //         EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
    //     }
        
    //     TYPED_TEST(QuantizedIntReshapeTest, InvalidQuantizationDimensionRaiseAnError) {
    //         using StorageT = typename TypeParam::StorageT;
    //         using ExpressedT = typename TypeParam::ExpressedT;

    //         const Shape shape_operand({2, 3, 2});
    //         const Shape shape_r({3, 2, 2});
    //         Vector<StorageT> operand_data =
    //             Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    //         Vector<StorageT> output_data(shape_r.NumElements());
    //         std::initializer_list<float> zero_points = {0, 0};
    //         std::initializer_list<float> scales = {1.2, 1.1};
    //         std::vector<int> zeroes = {0, 0};
    //         std::vector<float> scalesv = {1.2, 1.1};
    //         QuantizedElementTypePerAxis tensor_type_axis(
    //             TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);

    //         Tensor operand{
    //             .type = QuantizedPerAxisTensorType{.shape = shape_operand,
    //                                                 .element_type = tensor_type_axis},
    //             .data = operand_data.data()};
    //         Tensor output_tensor{
    //             .type = QuantizedPerAxisTensorType{.shape = shape_r,
    //                                                 .element_type = tensor_type_axis},
    //             .data = output_data.data()};

    //         auto op = Create(reshapeOp::Attributes{});

    //         const absl::Status status = Prepare(op, operand, output_tensor);
    //         EXPECT_THAT(status, shlo_ref::testing::StatusIs(
    //                                 absl::StatusCode::kFailedPrecondition));
    //         EXPECT_THAT(
    //             status.message(),
    //             "stablehlo.reshape: The quantization dimension of operand should be "
    //             "different to the quantization dimension of output.");
    //     }
    //     TYPED_TEST(QuantizedIntReshapeTest, DifferentElementTypeRaiseAnError) {
    //     using StorageT = typename TypeParam::StorageT;
    //     using ExpressedT = typename TypeParam::ExpressedT;

    //     const Shape shape_operand({2, 3, 2});
    //     const Shape shape_r({3, 2, 2});
    //     Vector<StorageT> operand_data =
    //         Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    //     Vector<StorageT> output_data(shape_r.NumElements());
    //     std::initializer_list<float> zero_points = {0, 0};
    //     std::initializer_list<float> scales = {1.2, 1.1};
    //     std::initializer_list<float> zero_points_output = {1, 0, 0};
    //     std::initializer_list<float> scales_output = {2, 1.2, 1.1};
    //     std::vector<int> zeroes = {0, 0};
    //     std::vector<float> scalesv = {1.2, 1.1};
    //     QuantizedElementTypePerAxis tensor_type_axis(
    //         TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);
    //     QuantizedElementTypePerAxis tensor_type_axis_output(
    //         TypeParam::kStorage, zero_points_output, TypeParam::kExpressed,
    //         scales_output, 1);
            
    //     Tensor operand{
    //         .type = QuantizedPerAxisTensorType{.shape = shape_operand,
    //                                             .element_type = tensor_type_axis},
    //         .data = operand_data.data()};
    //     Tensor output_tensor{
    //         .type =
    //             QuantizedPerAxisTensorType{.shape = shape_r,
    //                                         .element_type = tensor_type_axis_output},
    //         .data = output_data.data()};

    //     auto op = Create(reshapeOp::Attributes{});

    //     const absl::Status status = Prepare(op, operand, output_tensor);
    //     EXPECT_THAT(status, shlo_ref::testing::StatusIs(
    //                             absl::StatusCode::kFailedPrecondition));
    //     EXPECT_THAT(
    //         status.message(),::testing::ContainsRegex(
    //             "stablehlo.reshape: element type constraint is not satisfied"));
    //     }

        
    }

}