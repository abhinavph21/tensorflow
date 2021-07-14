# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for XLA op wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class XlaOpsNumericalTest(xla_test.XLATestCase, parameterized.TestCase):

  def _assertOpOutputMatchesExpected(self, op, args, expected,
                                     equality_fn=None):
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
      result = session.run(output, feeds)
      if not equality_fn:
        equality_fn = self.assertAllClose
      equality_fn(result, expected, rtol=1e-3)

  def testAdd(self):
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          xla.add,
          args=(np.array([1, 2, 3], dtype=dtype),
                np.array([4, 5, 6], dtype=dtype)),
          expected=np.array([5, 7, 9], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          lambda x, y: xla.add(x, y, broadcast_dims=(0,)),
          args=(np.array([[1, 2], [3, 4]], dtype=dtype),
                np.array([7, 11], dtype=dtype)),
          expected=np.array([[8, 9], [14, 15]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          lambda x, y: xla.add(x, y, broadcast_dims=(1,)),
          args=(np.array([[1, 2], [3, 4]], dtype=dtype),
                np.array([7, 11], dtype=dtype)),
          expected=np.array([[8, 13], [10, 15]], dtype=dtype))

  def testBroadcast(self):
    for dtype in self.numeric_types:
      v = np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2])
      self._assertOpOutputMatchesExpected(
          lambda x: xla.broadcast(x, (7, 42)),
          args=(v,),
          expected=np.tile(v, (7, 42, 1, 1)))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testGather(self):
    operand = np.arange(10, dtype=np.int32).reshape([2, 5])
    start_indices = np.array([2], np.int32)
    slice_sizes = np.array([1, 3], np.int32)

    def gather(operand, start_indices):
      dimension_numbers = xla_data_pb2.GatherDimensionNumbers()
      dimension_numbers.offset_dims.extend([1])
      dimension_numbers.collapsed_slice_dims.extend([0])
      dimension_numbers.start_index_map.extend([0])
      dimension_numbers.index_vector_dim = 1
      return xla.gather(operand, start_indices, dimension_numbers, slice_sizes)

    self._assertOpOutputMatchesExpected(
        gather,
        args=(operand, start_indices),
        expected=np.array([[5, 6, 7]]))

  def testShiftRightLogical(self):
    self._assertOpOutputMatchesExpected(
        xla.shift_right_logical,
        args=(np.array([-1, 16], dtype=np.int32), np.int32(4)),
        expected=np.array([0x0FFFFFFF, 1], dtype=np.int32))

    self._assertOpOutputMatchesExpected(
        xla.shift_right_logical,
        args=(np.array([0xFFFFFFFF, 16], dtype=np.uint32), np.uint32(4)),
        expected=np.array([0x0FFFFFFF, 1], dtype=np.uint32))

  def testShiftRightArithmetic(self):
    self._assertOpOutputMatchesExpected(
        xla.shift_right_arithmetic,
        args=(np.array([-1, 16], dtype=np.int32), np.int32(4)),
        expected=np.array([-1, 1], dtype=np.int32))

    self._assertOpOutputMatchesExpected(
        xla.shift_right_arithmetic,
        args=(np.array([0xFFFFFFFF, 16], dtype=np.uint32), np.uint32(4)),
        expected=np.array([0xFFFFFFFF, 1], dtype=np.uint32))

  PRECISION_VALUES = (None, xla_data_pb2.PrecisionConfig.DEFAULT,
                      xla_data_pb2.PrecisionConfig.HIGH,
                      xla_data_pb2.PrecisionConfig.HIGHEST)

  @parameterized.parameters(*PRECISION_VALUES)
  def testConv(self, precision):
    for dtype in set(self.float_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      def conv_1d_fn(lhs, rhs):
        dnums = xla_data_pb2.ConvolutionDimensionNumbers()
        num_spatial_dims = 1
        dnums.input_batch_dimension = 0
        dnums.input_feature_dimension = 1
        dnums.output_batch_dimension = 0
        dnums.output_feature_dimension = 1
        dnums.kernel_output_feature_dimension = 0
        dnums.kernel_input_feature_dimension = 1
        dnums.input_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        dnums.kernel_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        dnums.output_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        precision_config = None
        if precision:
          precision_config = xla_data_pb2.PrecisionConfig()
          precision_config.operand_precision.extend([precision, precision])
        return xla.conv(
            lhs,
            rhs,
            window_strides=(1,),
            padding=((2, 1),),
            lhs_dilation=(1,),
            rhs_dilation=(2,),
            dimension_numbers=dnums,
            precision_config=precision_config)

      self._assertOpOutputMatchesExpected(
          conv_1d_fn,
          args=(
              np.array([[[3, 4, 5, 6]]], dtype=dtype),
              np.array([[[-2, -3]]], dtype=dtype),
          ),
          expected=np.array([[[-9, -12, -21, -26, -10]]], dtype=dtype))

  def testConvPreferredElementType(self):
    dtype = np.float16
    preferred_element_type = np.float32

    def conv_1d_fn(lhs, rhs):
      dnums = xla_data_pb2.ConvolutionDimensionNumbers()
      num_spatial_dims = 1
      dnums.input_batch_dimension = 0
      dnums.input_feature_dimension = 1
      dnums.output_batch_dimension = 0
      dnums.output_feature_dimension = 1
      dnums.kernel_output_feature_dimension = 0
      dnums.kernel_input_feature_dimension = 1
      dnums.input_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
      dnums.kernel_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
      dnums.output_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
      precision_config = None
      return xla.conv(
          lhs,
          rhs,
          window_strides=(1,),
          padding=((2, 1),),
          lhs_dilation=(1,),
          rhs_dilation=(2,),
          dimension_numbers=dnums,
          precision_config=precision_config,
          preferred_element_type=preferred_element_type)

    self._assertOpOutputMatchesExpected(
        conv_1d_fn,
        args=(
            np.array([[[3, 4, 5, 6]]], dtype=dtype),
            np.array([[[-2, -3]]], dtype=dtype),
        ),
        expected=np.array([[[-9, -12, -21, -26, -10]]],
                          dtype=preferred_element_type))

  @parameterized.parameters(*PRECISION_VALUES)
  def testDotGeneral(self, precision):
    for dtype in self.float_types:

      def dot_fn(lhs, rhs):
        dnums = xla_data_pb2.DotDimensionNumbers()
        dnums.lhs_contracting_dimensions.append(2)
        dnums.rhs_contracting_dimensions.append(1)
        dnums.lhs_batch_dimensions.append(0)
        dnums.rhs_batch_dimensions.append(0)
        precision_config = None
        if precision:
          precision_config = xla_data_pb2.PrecisionConfig()
          precision_config.operand_precision.extend([precision, precision])
        return xla.dot_general(
            lhs,
            rhs,
            dimension_numbers=dnums,
            precision_config=precision_config)

      lhs = np.array(
          [
              [[1, 2], [3, 4]],
              [[5, 6], [7, 8]],
          ], dtype=dtype)
      rhs = np.array(
          [
              [[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]],
          ], dtype=dtype)
      self._assertOpOutputMatchesExpected(
          dot_fn,
          args=(lhs, rhs),
          expected=np.array(
              [
                  [[9, 12, 15], [19, 26, 33]],
                  [[95, 106, 117], [129, 144, 159]],
              ],
              dtype=dtype))

  def testDotGeneralInt8xInt8ToInt32(self):

    def dot_fn(lhs, rhs):
      dnums = xla_data_pb2.DotDimensionNumbers()
      dnums.lhs_contracting_dimensions.append(2)
      dnums.rhs_contracting_dimensions.append(1)
      dnums.lhs_batch_dimensions.append(0)
      dnums.rhs_batch_dimensions.append(0)
      return xla.dot_general(
          lhs, rhs, dimension_numbers=dnums, preferred_element_type=np.int32)

    lhs = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ], dtype=np.int8)
    rhs = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ],
                   dtype=np.int8)
    self._assertOpOutputMatchesExpected(
        dot_fn,
        args=(lhs, rhs),
        expected=np.array([
            [[9, 12, 15], [19, 26, 33]],
            [[95, 106, 117], [129, 144, 159]],
        ],
                          dtype=np.int32))

  def testNeg(self):
    for dtype in self.numeric_types - {np.uint8, np.int8}:
      self._assertOpOutputMatchesExpected(
          xla.neg,
          args=(np.array([1, 2, 3], dtype=dtype),),
          expected=np.array([-1, -2, -3], dtype=dtype))

  def testPad(self):
    for dtype in self.numeric_types:

      def pad_fn(x):
        return xla.pad(
            x,
            padding_value=7,
            padding_low=[2, 1],
            padding_high=[1, 2],
            padding_interior=[1, 0])

      self._assertOpOutputMatchesExpected(
          pad_fn,
          args=(np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2]),),
          expected=np.array(
              [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 0, 1, 7, 7],
               [7, 7, 7, 7, 7], [7, 2, 3, 7, 7], [7, 7, 7, 7, 7]],
              dtype=dtype))

  def testPadNegative(self):
    for dtype in self.numeric_types:

      def pad_fn(x):
        return xla.pad(
            x,
            padding_value=7,
            padding_low=[0, -1],
            padding_high=[1, -2],
            padding_interior=[1, 2])

      self._assertOpOutputMatchesExpected(
          pad_fn,
          args=(np.arange(6, dtype=np.int32).astype(dtype).reshape([2, 3]),),
          expected=np.array(
              [[7, 7, 1, 7], [7, 7, 7, 7], [7, 7, 4, 7], [7, 7, 7, 7]],
              dtype=dtype))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testReduce(self):
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      @function.Defun(dtype, dtype)
      def sum_reducer(x, y):
        return x + y

      def sum_reduction(dims):

        def fn(x):
          return xla.reduce(
              x, init_value=0, dimensions_to_reduce=dims, reducer=sum_reducer)

        return fn

      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]))
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[0]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([12, 15, 18, 21], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[1]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([6, 22, 38], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[0, 1]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=dtype(66))

      @function.Defun(dtype, dtype)
      def mul_reducer(x, y):
        return x * y

      def mul_reduction(dims):

        def fn(x):
          return xla.reduce(
              x, init_value=1, dimensions_to_reduce=dims, reducer=mul_reducer)

        return fn

      self._assertOpOutputMatchesExpected(
          mul_reduction(dims=[0]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([0, 45, 120, 231], dtype=dtype))

  @parameterized.parameters(False, True)
  @test_util.disable_mlir_bridge('Not supported yet')
  def testVariadicReduceKahanSum(self, use_v2):
    for dtype in set(self.numeric_types).intersection(
        set([np.float32, np.complex64])):

      @def_function.function
      def kahan_sum_reducer(t0, t1):
        (s0, c0), (s1, c1) = t0, t1
        s0minusc = s0 - (c0 + c1)
        t = s1 + s0minusc
        c = (t - s1) - s0minusc
        s = t
        return s, c

      def kahan_sum_reduction(dims, output_idx):

        def fn(x):
          arg = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
          reducer = kahan_sum_reducer.get_concrete_function(
              (arg, arg), (arg, arg))

          if use_v2:
            return xla.variadic_reduce_v2((x, array_ops.zeros_like(x)),
                                          init_values=(arg, arg),
                                          dimensions_to_reduce=dims,
                                          reducer=reducer)[output_idx]
          else:
            return xla.variadic_reduce((x, array_ops.zeros_like(x)),
                                       init_value=(arg, arg),
                                       dimensions_to_reduce=dims,
                                       reducer=reducer)[output_idx]

        return fn

      xs = np.array([1e5, np.pi, -1e5, np.exp(1.)])
      xs = np.array([xs, xs[::-1] / 3, xs / 7], dtype)
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[], output_idx=0),
          args=(xs,), expected=xs)
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[], output_idx=1),
          args=(xs,), expected=np.zeros_like(xs))
      shuffle_indices = np.argsort(np.random.randn(xs.shape[0]))
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0], output_idx=0),
          args=(xs[shuffle_indices],),
          expected=np.array([np.exp(1) / 3 + 1e5 * 8 / 7,
                             np.pi * 8 / 7 - 1e5 / 3,
                             -1e5 * 8 / 7 + np.pi / 3,
                             np.exp(1) * 8 / 7 + 1e5 / 3], dtype=dtype))
      error_term_equality = functools.partial(self.assertAllClose, atol=.005)
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0], output_idx=1),
          args=(xs[shuffle_indices],), expected=np.zeros_like(xs[0]),
          equality_fn=error_term_equality)
      shuffle_indices = np.argsort(np.random.randn(xs.shape[1]))
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[1], output_idx=0),
          args=(xs[:, shuffle_indices],),
          expected=np.array([np.pi + np.exp(1.),
                             (np.pi + np.exp(1.)) / 3,
                             (np.pi + np.exp(1.)) / 7], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[1], output_idx=1),
          args=(xs[:, shuffle_indices],), expected=np.zeros_like(xs[:, 0]),
          equality_fn=error_term_equality)
      # Now, shuffle both dims.
      xs = xs[np.argsort(np.random.randn(xs.shape[0]))]
      xs = xs[:, np.argsort(np.random.randn(xs.shape[1]))]
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0, 1], output_idx=0),
          args=(xs,), expected=dtype((np.pi + np.exp(1.)) * 31 / 21))
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0, 1], output_idx=1),
          args=(xs,), expected=dtype(0),
          equality_fn=error_term_equality)

  @test_util.disable_mlir_bridge('Not supported yet')
  def testVariadicReduceV2SingleOp(self):

    @def_function.function
    def reducer_add(op_element, acc_val):
      return (op_element + acc_val,)

    for dtype in set(self.numeric_types):
      values = np.array([[1, 3, 5], [4, 6, 8]], dtype=dtype)
      init_val = np.array(0, dtype=dtype)
      arg_spec = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
      reducer_func = reducer_add.get_concrete_function(arg_spec, arg_spec)

      def reduce(values, *, dimensions_to_reduce):
        return xla.variadic_reduce_v2((values,), (init_val,),  # pylint: disable=cell-var-from-loop
                                      dimensions_to_reduce=dimensions_to_reduce,
                                      reducer=reducer_func)[0]  # pylint: disable=cell-var-from-loop
      # Reduce dimension 0
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0,)),
          args=(values,),
          expected=np.array([5, 9, 13], dtype=dtype))

      # Reduce dimension 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(1,)),
          args=(values,),
          expected=np.array([9, 18], dtype=dtype))

      # Reduce dimensions 0 and 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0, 1)),
          args=(values,),
          expected=np.array(27, dtype=dtype))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testVariadicReduceV2DifferentTypes(self):
    # Two ops, with different dtypes
    @def_function.function
    def reducer_add(op_element_1, op_element_2, acc_val_1, acc_val_2):
      return (op_element_1 + acc_val_1, op_element_2 + acc_val_2)

    for dtype in set(self.numeric_types):
      values_1 = np.array([[1, 3, 5], [4, 6, 8]], dtype=dtype)
      values_2 = values_1.astype(np.int32)

      init_val_1 = np.array(0, dtype=dtype)  # pylint: disable=cell-var-from-loop
      init_val_2 = init_val_1.astype(np.int32)

      arg_spec_1 = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
      arg_spec_2 = array_ops.zeros([], np.int32)
      reducer_func = reducer_add.get_concrete_function(arg_spec_1, arg_spec_2,
                                                       arg_spec_1, arg_spec_2)  # pylint: disable=cell-var-from-loop

      def reduce(*values, dimensions_to_reduce):
        return xla.variadic_reduce_v2(values, (init_val_1, init_val_2,),  # pylint: disable=cell-var-from-loop
                                      dimensions_to_reduce=dimensions_to_reduce,
                                      reducer=reducer_func)  # pylint: disable=cell-var-from-loop

      # Reduce dimension 0
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0,)),
          args=(values_1, values_2),
          expected=(np.array([5, 9, 13],
                             dtype=dtype), np.array([5, 9, 13],
                                                    dtype=np.int32)))

      # Reduce dimension 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(1,)),
          args=(values_1, values_2),
          expected=(np.array([9, 18],
                             dtype=dtype), np.array([9, 18], dtype=np.int32)))

      # Reduce dimensions 0 and 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0, 1)),
          args=(values_1, values_2),
          expected=(np.array(27, dtype=dtype), np.array(27, dtype=np.int32)))

      # Reduce not dimensions
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=()),
          args=(values_1, values_2),
          expected=(values_1, values_2))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testSelectAndScatter(self):
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      @function.Defun(dtype, dtype)
      def add_scatter(x, y):
        return x + y

      @function.Defun(dtype, dtype)
      def ge_select(x, y):
        return x >= y

      def test_fn(operand, source):
        return xla.select_and_scatter(
            operand,
            window_dimensions=[2, 3, 1, 1],
            window_strides=[2, 2, 1, 1],
            padding=[[0, 0]] * 4,
            source=source,
            init_value=0,
            select=ge_select,
            scatter=add_scatter)

      self._assertOpOutputMatchesExpected(
          test_fn,
          args=(np.array(
              [[7, 2, 5, 3, 8], [3, 8, 9, 3, 4], [1, 5, 7, 5, 6],
               [0, 6, 2, 10, 2]],
              dtype=dtype).reshape((4, 5, 1, 1)),
                np.array([[2, 6], [3, 1]], dtype=dtype).reshape((2, 2, 1, 1))),
          expected=np.array(
              [[0, 0, 0, 0, 0], [0, 0, 8, 0, 0], [0, 0, 3, 0, 0],
               [0, 0, 0, 1, 0]],
              dtype=dtype).reshape((4, 5, 1, 1)))

  def testTranspose(self):
    for dtype in self.numeric_types:
      v = np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2])
      self._assertOpOutputMatchesExpected(
          lambda x: xla.transpose(x, [1, 0]), args=(v,), expected=v.T)

  def testDynamicSlice(self):
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          xla.dynamic_slice,
          args=(np.arange(1000,
                          dtype=np.int32).astype(dtype).reshape([10, 10, 10]),
                np.array([5, 7, 3]), np.array([2, 3, 2])),
          expected=np.array(
              np.array([[[573, 574], [583, 584], [593, 594]],
                        [[673, 674], [683, 684], [693, 694]]]),
              dtype=dtype))

  @test_util.disable_mlir_bridge('Error handling')
  def testDynamicSliceWithIncorrectStartIndicesShape(self):
    with self.session() as session:
      with self.test_scope():
        output = xla.dynamic_slice(
            np.arange(1000, dtype=np.int32).reshape([10, 10, 10]),
            np.array([5, 7]), np.array([2, 3, 4]))
      with self.assertRaises(errors.InvalidArgumentError) as invalid_arg_error:
        session.run(output)
      self.assertRegex(
          invalid_arg_error.exception.message,
          (r'start_indices must be a vector with length equal to input rank, '
           r'but input rank is 3 and start_indices has shape \[2\].*'))

  @test_util.disable_mlir_bridge('Error handling')
  def testDynamicSliceWithIncorrectSizeIndicesShape(self):
    with self.session() as session:
      with self.test_scope():
        output = xla.dynamic_slice(
            np.arange(1000, dtype=np.int32).reshape([10, 10, 10]),
            np.array([5, 7, 3]), np.array([2, 3]))
      with self.assertRaises(errors.InvalidArgumentError) as invalid_arg_error:
        session.run(output)
      self.assertRegex(
          invalid_arg_error.exception.message,
          (r'size_indices must be a vector with length equal to input rank, '
           r'but input rank is 3 and size_indices has shape \[2\].*'))


class XlaOpsShapeInferenceTest(xla_test.XLATestCase, parameterized.TestCase):

  def testDotShapeInference(self):
    a = array_ops.placeholder(np.float32, shape=(1, 2, 3, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 5, 2, 6))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(2)
    dim_nums.lhs_batch_dimensions.append(3)
    dim_nums.rhs_batch_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape, tensor_shape.TensorShape([4, 1, 3, 5, 6]))

  def testDotDifferentNumberOfContractingDimensions(self):
    a = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)

    with self.assertRaisesRegex(ValueError,
                                'Must specify the same number of contracting '
                                'dimensions for lhs and rhs. Got: 1 and 2'):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentContractingDimensionsSizes(self):
    a = array_ops.placeholder(np.float32, shape=(2, 2, 2, 2))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)

    with self.assertRaisesRegex(ValueError,
                                'Dimensions must be equal, but are 2 and 4'):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentNumberOfBatchDimensions(self):
    a = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_batch_dimensions.append(2)
    dim_nums.rhs_batch_dimensions.append(2)
    dim_nums.rhs_batch_dimensions.append(3)

    with self.assertRaisesRegex(ValueError,
                                'Must specify the same number of batch '
                                'dimensions for lhs and rhs. Got: 1 and 2'):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentBatchDimensionsSizes(self):
    a = array_ops.placeholder(np.float32, shape=(2, 2, 2, 2))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)
    dim_nums.lhs_batch_dimensions.append(0)
    dim_nums.rhs_batch_dimensions.append(0)

    with self.assertRaisesRegex(ValueError,
                                'Dimensions must be equal, but are 2 and 4'):
      xla.dot_general(a, b, dim_nums)

  def testDotUnknownNonContractingDimension(self):
    a = array_ops.placeholder(np.float32, shape=(None, 16))
    b = array_ops.placeholder(np.float32, shape=(16, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [None, 2])

  def testDotUnknownContractingDimension(self):
    a = array_ops.placeholder(np.float32, shape=(3, None))
    b = array_ops.placeholder(np.float32, shape=(None, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [3, 2])

  def testDotUnknownAndKnownContractingDimension(self):
    a = array_ops.placeholder(np.float32, shape=(3, 4))
    b = array_ops.placeholder(np.float32, shape=(None, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [3, 2])

  def testDotUnknownBatchDimension(self):
    a = array_ops.placeholder(np.float32, shape=(None, 3, 4))
    b = array_ops.placeholder(np.float32, shape=(None, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(1)
    dim_nums.lhs_batch_dimensions.append(0)
    dim_nums.rhs_batch_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [None, 3])

  def testDotUnknownAndKnownBatchDimension(self):
    a = array_ops.placeholder(np.float32, shape=(2, 3, 4))
    b = array_ops.placeholder(np.float32, shape=(None, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(1)
    dim_nums.lhs_batch_dimensions.append(0)
    dim_nums.rhs_batch_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [2, 3])

  def testPadShapeInference(self):
    a = array_ops.placeholder(np.float32, shape=(2, 3))

    c = xla.pad(
        a,
        padding_value=7,
        padding_low=[2, 1],
        padding_high=[1, 2],
        padding_interior=[1, 4])

    self.assertEqual(c.shape, tensor_shape.TensorShape([6, 14]))

    c = xla.pad(
        a,
        padding_value=7,
        padding_low=[2, -2],
        padding_high=[1, -2],
        padding_interior=[1, 2])

    self.assertEqual(c.shape, tensor_shape.TensorShape([6, 3]))

    c = xla.pad(
        array_ops.placeholder(np.float32, shape=(None, 2)),
        padding_value=7,
        padding_low=[0, 1],
        padding_high=[0, 2],
        padding_interior=[0, 4])
    self.assertEqual(c.shape.as_list(), [None, 9])

    # 0-sized input dimension and interior padding
    c = xla.pad(
        array_ops.placeholder(np.float32, shape=(2, 0)),
        padding_value=7,
        padding_low=[2, 1],
        padding_high=[1, 1],
        padding_interior=[1, 2])

    self.assertEqual(c.shape, tensor_shape.TensorShape([6, 2]))

    with self.assertRaisesRegex(
        ValueError, 'padding_value input must be scalar, found rank 1 '):
      xla.pad(
          a,
          padding_value=[0, 1],
          padding_low=[0, 0],
          padding_high=[0, 0],
          padding_interior=[0, 0])

    with self.assertRaisesRegex(ValueError,
                                'padding_low must be a 1D tensor of size 2 '):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0, 0],
          padding_high=[0, 0],
          padding_interior=[0, 0])

    with self.assertRaisesRegex(ValueError,
                                'padding_high must be a 1D tensor of size 2 '):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0],
          padding_high=[0, 0, 0],
          padding_interior=[0, 0])

    with self.assertRaisesRegex(
        ValueError, 'padding_interior must be a 1D tensor of size 2 '):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0],
          padding_high=[0, 0],
          padding_interior=[0])

    with self.assertRaisesRegex(
        ValueError,
        'padding_interior must contain only non-negative values, found -2 '):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0],
          padding_high=[0, 0],
          padding_interior=[-2, 0])

    with self.assertRaisesRegex(
        ValueError, 'resulting padded dimension has negative size -1 '):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[-3, 0],
          padding_high=[0, 0],
          padding_interior=[0, 0])

  def testVariadicReduceV2SingleArg(self):

    @def_function.function
    def reducer_add(op_element, acc_val):
      return (op_element + acc_val,)

    dtype = np.float32
    arg_spec = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
    reducer_func = reducer_add.get_concrete_function(arg_spec, arg_spec)

    res = xla.variadic_reduce_v2(
        (array_ops.placeholder(np.float32, shape=(3, 4, 5)),),
        (array_ops.placeholder(np.float32, shape=()),),
        dimensions_to_reduce=(1,),
        reducer=reducer_func)
    self.assertLen(res, 1)
    self.assertEqual(res[0].shape, tensor_shape.TensorShape([3, 5]))

  def testVariadicReduceV2MultipleArgs(self):

    @def_function.function
    def reducer_adds(op_element_1, op_element_2, op_element_3, acc_val_1,
                     acc_val_2, acc_val_3):
      return (op_element_1 + acc_val_1, op_element_2 + acc_val_2,
              op_element_3 + acc_val_3)

    dtype = np.float32
    arg1_spec = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
    arg2_spec = array_ops.zeros([], np.int32)
    arg3_spec = array_ops.zeros([], np.int32)
    reducer_func = reducer_adds.get_concrete_function(arg1_spec, arg2_spec,
                                                      arg3_spec, arg1_spec,
                                                      arg2_spec, arg3_spec)

    def reduce_with_shapes(shape1, shape2, shape3, dimensions_to_reduce=(1,)):
      inputs = (array_ops.placeholder(np.float32, shape=shape1),
                array_ops.placeholder(np.int32, shape=shape2),
                array_ops.placeholder(np.int32, shape=shape3))
      init_values = (array_ops.placeholder(np.float32, shape=()),
                     array_ops.placeholder(np.int32, shape=()),
                     array_ops.placeholder(np.int32, shape=()))

      return xla.variadic_reduce_v2(
          inputs,
          init_values,
          dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer_func)

    def assert_output_shapes(output, expected_shape):
      self.assertLen(output, 3)
      self.assertEqual(output[0].shape.as_list(), list(expected_shape))
      self.assertEqual(output[1].shape.as_list(), list(expected_shape))
      self.assertEqual(output[2].shape.as_list(), list(expected_shape))

    output = reduce_with_shapes((3, 4, 5), (3, 4, 5), (3, 4, 5))
    assert_output_shapes(output, (3, 5))

    output = reduce_with_shapes((3, 4, 5), (3, 4, 5), (3, 4, 5),
                                dimensions_to_reduce=())
    assert_output_shapes(output, (3, 4, 5))

    output = reduce_with_shapes(None, (3, None, 5), (None, 4, 5))
    assert_output_shapes(output, (3, 5))

    output = reduce_with_shapes(None, (3, None, 5), None)
    assert_output_shapes(output, (3, 5))

    output = reduce_with_shapes(None, (None, None, 5), None)
    assert_output_shapes(output, (None, 5))

    output = reduce_with_shapes(None, None, None)
    self.assertLen(output, 3)
    self.assertIsNone(output[0].shape.rank)
    self.assertIsNone(output[1].shape.rank)
    self.assertIsNone(output[2].shape.rank)

    with self.assertRaisesRegex(ValueError,
                                'All inputs must have the same shape'):
      reduce_with_shapes((3, 4, 5), (13, 4, 5), (3, 4, 5))

    with self.assertRaisesRegex(ValueError,
                                'All inputs must have the same shape'):
      reduce_with_shapes((None, 4, 5), (3, None, 5), (13, 4, 5))

    with self.assertRaisesRegex(ValueError,
                                'All inputs must have the same shape'):
      reduce_with_shapes((None, 4, 5), (3, None, 5), (13, 4, 5))


if __name__ == '__main__':
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  googletest.main()
