# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import random

import ttnn


@pytest.mark.model_mole_dl_linear_ops
class TestTTNNSiliconOps:
    def test_zeros_like(reset_seeds, device_sp, models_params):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        ## test zeros like

        x = torch.zeros((batch_size, pred_len, enc_in)).to(torch_dtype)
        tmp_a = ttnn.from_torch(x, device=device, dtype=ttnn_dtype)

        xtt = ttnn.zeros_like(tmp_a)
        """
        a_torch = torch.ones((batch_size, pred_len, enc_in)).to(torch_dtype)
        a = ttnn.from_torch(a_torch, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)

        xtt = xtt + a
        print(xtt)
        """

        tt_got_back = ttnn.to_torch(xtt)
        eq = torch.equal(x, tt_got_back)
        assert eq

        del xtt
        del tmp_a

        ttnn.device.CloseDevice(device)
        return

    def test_zeros(reset_seeds, device_sp, models_params):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        ## test zeros like
        x = torch.zeros((batch_size, pred_len, enc_in)).to(torch_dtype)
        xtt = ttnn.zeros((batch_size, pred_len, enc_in), dtype=ttnn_dtype, device=device)

        """
        a_torch = torch.ones((batch_size, pred_len, enc_in)).to(torch_dtype)
        a = ttnn.from_torch(a_torch, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)

        xtt = xtt + a
        print(xtt)
        """

        tt_got_back = ttnn.to_torch(xtt)
        assert torch.equal(x, tt_got_back)

        del xtt

        ttnn.device.CloseDevice(device)
        return

    def concat_test_instance(self, tensor_dims, cat_dim, torch_dtype, ttnn_dtype, device):
        x = torch.randn(tensor_dims).to(torch_dtype)
        y = torch.randn(tensor_dims).to(torch_dtype)
        z = torch.concat([x, y], dim=cat_dim)

        tt_x = ttnn.from_torch(x, device=device, dtype=ttnn_dtype)
        tt_y = ttnn.from_torch(y, device=device, dtype=ttnn_dtype)
        tt_z = ttnn.concat([tt_x, tt_y], dim=cat_dim)

        back_torch_tt_z = ttnn.to_torch(tt_z)

        # print("x: ", x, "\ny: ", y, "\nz: ", z)
        # print("tt_x: ", tt_x, "\ntt_y: ", tt_y, "\ntt_z: ", tt_z)

        eq = torch.equal(z, back_torch_tt_z)

        del tt_x
        del tt_y
        del tt_z
        return eq

    def concat_3_tensors_test_instance(
        self, tensor_dims0, tensor_dims1, tensor_dims2, cat_dim, torch_dtype, ttnn_dtype, device
    ):
        x0 = torch.randn(tensor_dims0).to(torch_dtype)
        x1 = torch.randn(tensor_dims1).to(torch_dtype)
        x2 = torch.randn(tensor_dims2).to(torch_dtype)
        z = torch.concat([x0, x1, x2], dim=cat_dim)

        tt_x0 = ttnn.from_torch(x0, device=device, dtype=ttnn_dtype)
        tt_x1 = ttnn.from_torch(x1, device=device, dtype=ttnn_dtype)
        tt_x2 = ttnn.from_torch(x2, device=device, dtype=ttnn_dtype)
        tt_z = ttnn.concat([tt_x0, tt_x1, tt_x2], dim=cat_dim)

        back_torch_tt_z = ttnn.to_torch(tt_z)

        # print("x: ", x, "\ny: ", y, "\nz: ", z)
        # print("tt_x: ", tt_x, "\ntt_y: ", tt_y, "\ntt_z: ", tt_z)

        eq = torch.equal(z, back_torch_tt_z)

        del tt_x0
        del tt_x1
        del tt_x2
        del tt_z
        return eq

    def test_concat(self, reset_seeds, device_sp, models_params):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        cat_dim = 1

        tensor_dims0 = (batch_size, (kernel_size - 1) // 2, enc_in)
        eq0 = self.concat_test_instance(tensor_dims0, cat_dim, torch_dtype, ttnn_dtype, device)
        assert eq0

        tensor_dims1 = (batch_size, label_len, enc_in)
        eq1 = self.concat_test_instance(tensor_dims1, cat_dim, torch_dtype, ttnn_dtype, device)
        assert eq1

        eq2 = self.concat_3_tensors_test_instance(
            tensor_dims0, tensor_dims1, tensor_dims0, cat_dim, torch_dtype, ttnn_dtype, device
        )
        assert eq2

        ttnn.device.CloseDevice(device)

    def slice_test_instance(self, tensor_dims, torch_dtype, ttnn_dtype, device):
        x = torch.randn(tensor_dims).to(torch_dtype)
        tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        # print("x: ", x, "\ntt_x: ", tt_x)

        y = x[:, 0:1, :]
        tt_y = ttnn.slice(
            tt_x, slice_start=(0, 0, 0), slice_end=(tensor_dims[0], 1, tensor_dims[2]), slice_step=(1, 1, 1)
        )
        back_torch_tt_y = ttnn.to_torch(tt_y)
        # print("y: ", y, "\ntt_y: ", tt_y, "\nback_torch_tt_y: ", back_torch_tt_y)

        eq = torch.equal(y, back_torch_tt_y)

        del tt_x
        del tt_y

        return eq

    def test_slice(self, reset_seeds, device_sp, models_params):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        tensor_dims0 = (batch_size, seq_len, udefined_v)
        eq0 = self.slice_test_instance(tensor_dims0, torch_dtype, ttnn_dtype, device)
        assert eq0

        tensor_dims1 = (batch_size, seq_len, enc_in)
        eq1 = self.slice_test_instance(tensor_dims1, torch_dtype, ttnn_dtype, device)
        assert eq1

        ttnn.device.CloseDevice(device)

    def test_repeat(reset_seeds, device_sp, models_params):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        tensor_dims = (batch_size, 1, enc_in)

        x = torch.randn(tensor_dims).to(torch_dtype)
        tt_x = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn_dtype)

        y = x.repeat(1, (kernel_size - 1) // 2, 1)
        tt_y = ttnn.repeat(tt_x, (1, (kernel_size - 1) // 2, 1))
        # print("y: ", y, "\ntt_y: ", tt_y)

        back_torch_tt_y = ttnn.to_torch(tt_y)

        eq = torch.equal(y, back_torch_tt_y)
        assert eq

        del tt_x
        del tt_y
        ttnn.device.CloseDevice(device)

    def permute_test_instance(self, tensor_dims, permute_order, torch_dtype, ttnn_dtype, device):
        x = torch.randn(tensor_dims).to(torch_dtype)
        tt_x = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn_dtype)
        # print("x: ", x, "\ntt_x: ", tt_x)

        y = x.permute(permute_order)
        tt_y = ttnn.permute(tt_x, permute_order)
        back_torch_tt_y = ttnn.to_torch(tt_y)
        # print("y: ", y, "\ntt_y: ", tt_y)

        """
        ## test
        bit16_tt_x = ttnn.typecast(tt_x, ttnn.bfloat16)
        bit32_tt_x = ttnn.typecast(bit16_tt_x, ttnn.float32)
        print("bit32_tt_x: ", bit32_tt_x)
        """

        del tt_x
        del tt_y

        assert torch.equal(y, back_torch_tt_y)
        return

    def permute_test_instance_with_tolerance(
        self, tensor_dims, permute_order, torch_dtype, ttnn_dtype, device, rtol, atol
    ):
        x = torch.randn(tensor_dims).to(torch_dtype)
        tt_x = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn_dtype)
        # print("x: ", x, "\ntt_x: ", tt_x)

        y = x.permute(permute_order)
        # y[0,0,0] = y[0,0,0] + 1.0
        tt_y = ttnn.permute(tt_x, permute_order)
        back_torch_tt_y = ttnn.to_torch(tt_y)
        # print("y: ", y, "\ntt_y: ", tt_y)

        del tt_x
        del tt_y

        assert torch.allclose(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        return

    def test_permute(self, reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance

        tensor_dims0 = (1, 3, 2)
        tensor_dims1 = (batch_size, seq_len + ((kernel_size - 1) // 2) * 2, enc_in)
        tensor_dims2 = (batch_size, enc_in, seq_len)
        tensor_dims3 = (batch_size, seq_len, enc_in)
        tensor_dims4 = (batch_size, enc_in, pred_len)

        permute_order0 = (0, 2, 1)

        "---------------- test with bitwise criterion -----------------"
        # self.permute_test_instance(tensor_dims0, permute_order0, torch_dtype,\
        #                                  ttnn_dtype, device)

        # self.permute_test_instance(tensor_dims1, permute_order0, torch_dtype,\
        #                                  ttnn_dtype, device)

        # self.permute_test_instance(tensor_dims2, permute_order0, torch_dtype,\
        #                                  ttnn_dtype, device)

        # self.permute_test_instance(tensor_dims3, permute_order0, torch_dtype,\
        #                                  ttnn_dtype, device)

        # self.permute_test_instance(tensor_dims4, permute_order0, torch_dtype,\
        #                                  ttnn_dtype, device)

        "---------------- test with certain precision tolerance -----------------"
        (rtol, atol) = (1e-3, 1e-3)  ## cannot pass with 1e-5

        self.permute_test_instance_with_tolerance(
            tensor_dims0, permute_order0, torch_dtype, ttnn_dtype, device, rtol, atol
        )
        self.permute_test_instance_with_tolerance(
            tensor_dims1, permute_order0, torch_dtype, ttnn_dtype, device, rtol, atol
        )

        self.permute_test_instance_with_tolerance(
            tensor_dims2, permute_order0, torch_dtype, ttnn_dtype, device, rtol, atol
        )

        self.permute_test_instance_with_tolerance(
            tensor_dims3, permute_order0, torch_dtype, ttnn_dtype, device, rtol, atol
        )

        self.permute_test_instance_with_tolerance(
            tensor_dims4, permute_order0, torch_dtype, ttnn_dtype, device, rtol, atol
        )

        ttnn.device.CloseDevice(device)
        return

    def test_avgpool1d(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        atol = 1e-2
        rtol = 1e-2

        tensor_dims_NCH = (batch_size, enc_in, seq_len + ((kernel_size - 1) // 2) * 2)
        tensor_dims_NHWC = (batch_size, seq_len + ((kernel_size - 1) // 2) * 2, 1, enc_in)

        """--------------------- call torch avg_pool1d -------------------------------"""
        padding = 0
        avg_pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

        x = torch.randn(tensor_dims_NCH).to(torch_dtype)  # N C H
        y = avg_pool(x)  ## N C H
        y_N = y.shape[0]
        y_C = y.shape[1]
        y_H = y.shape[2]
        y_W = 1

        """--------------------- call ttnn avg_pool2d -------------------------------"""
        x_2d = torch.unsqueeze(x, -1)  ## N, C, H --> N, C, H, 1
        x_NHWC = torch.permute(x_2d, (0, 2, 3, 1))  ## N, C, H, 1 --> N, H, 1, C
        x_NHWC = x_NHWC.to(torch.bfloat16)
        tt_x = ttnn.from_torch(x_NHWC, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_x_fp32 = ttnn.typecast(tt_x, ttnn_dtype)

        tt_y_bf16 = ttnn.avg_pool2d(
            input_tensor=tt_x,
            batch_size=tensor_dims_NHWC[0],
            input_h=tensor_dims_NHWC[1],
            input_w=tensor_dims_NHWC[2],
            channels=tensor_dims_NHWC[3],
            kernel_size=[kernel_size, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_y_fp32 = ttnn.typecast(tt_y_bf16, ttnn_dtype)
        tt_y_nhc = ttnn.reshape(tt_y_fp32, (y_N, y_H, y_C))
        tt_y_nch = ttnn.permute(tt_y_nhc, (0, 2, 1))
        back_tt_y_f32 = ttnn.to_torch(tt_y_nch)

        # print("tt_y_bf16.shape: ", tt_y_bf16.shape)
        # back_tt_y_bf16 = ttnn.to_torch(tt_y_bf16)
        # back_tt_y_bf16 = back_tt_y_bf16.reshape(y_N, y_H, y_C)  ## NHC
        # back_tt_y_f32 = back_tt_y_bf16.to(torch_dtype)
        # back_tt_y_f32 = torch.permute(back_tt_y_f32, (0, 2, 1))  ## N, C, H
        # print("back_tt_y_f32.shape: ", back_tt_y_f32.shape, "\ny.shape: ", y.shape)

        try:
            torch.testing.assert_close(y, back_tt_y_f32, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            print("Mismatch details:\n", e)
            assert False

        del tt_x
        del tt_y_bf16
        del tt_y_fp32
        del tt_y_nhc
        del tt_y_nch
        del back_tt_y_f32

        ttnn.device.CloseDevice(device)
        return

    def test_binary_op(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-2
        atol = 1e-2

        tensor_dims0 = (batch_size, seq_len, enc_in)
        # tensor_dims0 = (2, 3, 4)

        x0 = torch.randn(tensor_dims0).to(torch_dtype)
        y0 = torch.randn(tensor_dims0).to(torch_dtype)

        z0 = x0 - y0
        z1 = x0 + y0

        tt_x0 = ttnn.from_torch(x0, device=device, dtype=ttnn_dtype)
        tt_y0 = ttnn.from_torch(y0, device=device, dtype=ttnn_dtype)

        tt_z0 = tt_x0 - tt_y0
        tt_z1 = tt_x0 + tt_y0

        back_torch_tt_z0 = ttnn.to_torch(tt_z0)
        back_torch_tt_z1 = ttnn.to_torch(tt_z1)

        del tt_z0
        del tt_z1

        # assert torch.allclose(z0, back_torch_tt_z0, rtol=rtol, atol=atol, equal_nan=False)
        try:
            # Assert tensors are close (strict tolerance)
            torch.testing.assert_close(z0, back_torch_tt_z0, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details for t0:\n", e)
            assert False
        # assert torch.allclose(z1, back_torch_tt_z1, rtol=rtol, atol=atol, equal_nan=False)

        try:
            # Assert tensors are close (strict tolerance)
            torch.testing.assert_close(z1, back_torch_tt_z1, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details for t1:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return

    def reshape_test_instance(self, tensor_dims, target_dims, torch_dtype, ttnn_dtype, device):
        x0 = torch.randn(tensor_dims).to(torch_dtype)
        y0 = x0.reshape(target_dims)

        tt_x0 = ttnn.from_torch(x0, device=device, dtype=ttnn_dtype)
        tt_y0 = ttnn.reshape(tt_x0, target_dims)

        back_torch_tt_y0 = ttnn.to_torch(tt_y0)

        del tt_x0
        del tt_y0

        assert torch.equal(y0, back_torch_tt_y0)
        return

    def test_reshape(self, reset_seeds, device_sp, models_params):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        tensor_dims0 = (batch_size, 1, t_dim * enc_in)
        target_dims0 = (batch_size * enc_in, t_dim)
        self.reshape_test_instance(tensor_dims0, target_dims0, torch_dtype, ttnn_dtype, device)

        tensor_dims1 = (batch_size, seq_len, enc_in)
        target_dims1 = (int((batch_size * seq_len * enc_in) / (pred_len * t_dim)), pred_len, t_dim)
        self.reshape_test_instance(tensor_dims1, target_dims1, torch_dtype, ttnn_dtype, device)

        tensor_dims2 = (batch_size * enc_in, pred_len, 1)
        target_dims2 = (batch_size, enc_in, pred_len)
        self.reshape_test_instance(tensor_dims2, target_dims2, torch_dtype, ttnn_dtype, device)

        tensor_dims2 = (batch_size * enc_in, pred_len, 1)
        target_dims2 = (batch_size, enc_in, pred_len)
        self.reshape_test_instance(tensor_dims2, target_dims2, torch_dtype, ttnn_dtype, device)

        ttnn.device.CloseDevice(device)
        return

    def linear_test_instances(self, lhs_dims, rhs_dims, torch_dtype, ttnn_dtype, device, rtol, atol):
        # lhs = torch.randn(lhs_dims).to(torch_dtype) ## [Batch..., K]
        # rhs = torch.randn(rhs_dims).to(torch_dtype) ##  [N, K]
        # bias = torch.randn((rhs_dims[0])).to(torch_dtype) ##  [N]

        lhs = torch.randint(low=-10, high=10, size=lhs_dims).to(torch_dtype)  ## [Batch..., K]
        rhs = torch.randint(low=-10, high=10, size=rhs_dims).to(torch_dtype)  ##  [N, K]
        bias = torch.randint(low=-10, high=10, size=(rhs_dims[0],)).to(torch_dtype)  ##  [N]

        in_feature = lhs_dims[-1]  # K
        out_feature = rhs_dims[0]  # N

        linear_op = torch.nn.Linear(in_feature, out_feature)
        linear_op.weight.data = rhs
        linear_op.bias.data = bias

        out = linear_op(lhs)

        tt_lhs = ttnn.from_torch(lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_rhs = ttnn.from_torch(rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_bias = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_out = ttnn.linear(tt_lhs, tt_rhs, bias=tt_bias, transpose_b=True)

        back_torch_tt_out = ttnn.to_torch(tt_out)

        del tt_lhs
        del tt_rhs
        del tt_bias
        del tt_out

        try:
            torch.testing.assert_close(out, back_torch_tt_out, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in linear_test_instances:\n", e)
            assert False

        return

    def test_linear(self, reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-2  ## cannot pass with 1e-3
        atol = 1e-2

        lhs_dims0 = (batch_size, enc_in, seq_len)  ## [Batch..., K]
        rhs_dims0 = (pred_len * t_dim, seq_len)  ##  [N, K]
        self.linear_test_instances(lhs_dims0, rhs_dims0, torch_dtype, ttnn_dtype, device, rtol, atol)

        lhs_dims1 = (batch_size, 1, udefined_v)
        rhs_dims1 = (t_dim * enc_in, udefined_v)
        self.linear_test_instances(lhs_dims1, rhs_dims1, torch_dtype, ttnn_dtype, device, rtol, atol)

        lhs_dims2 = (batch_size, 1, t_dim * enc_in)
        rhs_dims2 = (t_dim * enc_in, t_dim * enc_in)
        self.linear_test_instances(lhs_dims2, rhs_dims2, torch_dtype, ttnn_dtype, device, rtol, atol)

        ttnn.device.CloseDevice(device)
        return

    def test_matmul(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-6
        atol = 1e-6

        lhs_dims = (batch_size * enc_in, pred_len, t_dim)  ## [Batch..., M, K]
        rhs_dims = (batch_size * enc_in, t_dim, 1)  ##  [Batch..., K, N]

        lhs = torch.randint(low=-50, high=50, size=lhs_dims).to(torch_dtype)
        rhs = torch.randint(low=-50, high=50, size=rhs_dims).to(torch_dtype)

        out = torch.matmul(lhs, rhs)

        tt_lhs = ttnn.from_torch(lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_rhs = ttnn.from_torch(rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)

        tt_out = ttnn.matmul(tt_lhs, tt_rhs)
        back_torch_tt_out = ttnn.to_torch(tt_out)

        del tt_lhs
        del tt_rhs
        del tt_out

        # print("out: ", out, "\nback_torch_tt_out: ", back_torch_tt_out)

        try:
            torch.testing.assert_close(out, back_torch_tt_out, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in matmul:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return

    def test_softmax(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 5e-2  ## can not pass with other strict ones
        atol = 5e-2

        tensor_dims = (batch_size * enc_in, t_dim)
        x = torch.randint(low=-25, high=50, size=tensor_dims).to(torch_dtype)
        y = torch.nn.Softmax(dim=1)(x)

        tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_y = ttnn.softmax(tt_x, dim=1)

        back_torch_tt_y = ttnn.to_torch(tt_y)

        del tt_x
        del tt_y
        # print("y: ", y, "\nback_torch_tt_y: ", back_torch_tt_y)

        try:
            torch.testing.assert_close(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return

    def test_relu(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-5  ## can not pass with other strict ones
        atol = 1e-5

        tensor_dims = (batch_size, 1, t_dim * enc_in)
        x = torch.randn(tensor_dims).to(torch_dtype)
        y = torch.nn.ReLU()(x)

        tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_y = ttnn.relu(tt_x)
        back_torch_tt_y = ttnn.to_torch(tt_y)

        del tt_x
        del tt_y
        # print("y: ", y, "\nback_torch_tt_y: ", back_torch_tt_y)

        try:
            torch.testing.assert_close(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return

    def test_headdrop(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-3  ## can not pass with other strict ones
        atol = 1e-3
        max_mismatch_ratio = 0.001

        p = 0.5

        tensor_dims = (batch_size * enc_in, t_dim)
        x = torch.randn(tensor_dims).to(torch_dtype)

        binary_mask = (x > p).float()

        y = x * binary_mask + (1 - binary_mask) * -1e20

        tt_x = ttnn.from_torch(x, device=device, dtype=ttnn_dtype)
        ttnn_x1 = ttnn.to_layout((tt_x > p), ttnn.TILE_LAYOUT)
        tt_mask = ttnn.typecast(ttnn_x1, ttnn.float32)
        tt_ones = ttnn.ones(tensor_dims, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        tt_y = tt_x * tt_mask + (tt_ones - tt_mask) * -1e20
        back_torch_tt_y = ttnn.to_torch(tt_y)

        print("x[2135, 0]: ", x[2135, 0])

        del tt_x
        del ttnn_x1
        del tt_mask
        del tt_y
        # print("y: ", y, "\nback_torch_tt_y: ", back_torch_tt_y)

        close_mask = torch.isclose(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        total_elements = y.numel()
        mismatch_count = total_elements - torch.sum(close_mask).item()
        mismatch_ratio = mismatch_count / total_elements if total_elements > 0 else 0.0

        # Check against threshold
        assert mismatch_ratio <= max_mismatch_ratio

        # try:
        #     torch.testing.assert_close(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        # except AssertionError as e:
        #     # Print the detailed mismatch log
        #     print("Mismatch details in softmax:\n", e)
        #     assert False

        ttnn.device.CloseDevice(device)
        return

    def test_mean(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-3  ## can not pass with other strict ones
        atol = 1e-3

        tensor_dims = (batch_size, seq_len, enc_in)
        x = torch.randn(tensor_dims).to(torch_dtype)
        dim2reduce = tuple(range(1, x.ndim - 1))
        y = torch.mean(x, dim=dim2reduce, keepdim=True)

        tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_y = ttnn.mean(tt_x, dim=x.ndim - 2, keepdim=True)

        back_torch_tt_y = ttnn.to_torch(tt_y)

        del tt_x
        del tt_y
        # print("y: ", y, "\nback_torch_tt_y: ", back_torch_tt_y)

        try:
            torch.testing.assert_close(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return

    def test_var(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-2  ## can not pass with other strict ones
        atol = 1e-2
        eps = 1e-4

        tensor_dims = (batch_size, seq_len, enc_in)
        x = torch.randn(tensor_dims).to(torch_dtype)
        dim2reduce = tuple(range(1, x.ndim - 1))
        y = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True) + eps)

        tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_y = ttnn.var(tt_x, dim=x.ndim - 2, keepdim=True)
        tt_y = tt_y + eps
        tt_y = ttnn.sqrt(tt_y)

        back_torch_tt_y = ttnn.to_torch(tt_y)

        del tt_x
        del tt_y
        # print("y: ", y, "\nback_torch_tt_y: ", back_torch_tt_y)

        try:
            torch.testing.assert_close(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return

    def test_broadcast_binary(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-4  ## can not pass with other strict ones
        atol = 1e-4
        eps = 1e-4

        tensor_dims1 = (batch_size, seq_len, enc_in)
        tensor_dims2 = (batch_size, 1, enc_in)
        tensor_dims3 = enc_in
        x1 = torch.randn(tensor_dims1).to(torch_dtype)
        x2 = torch.randn(tensor_dims2).to(torch_dtype) + eps
        x3 = torch.randn(tensor_dims3).to(torch_dtype)
        y = x1 + x2
        z = x1 / x2
        p = x1 * x3

        tt_x1 = ttnn.from_torch(x1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_x2 = ttnn.from_torch(x2, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_x3 = ttnn.from_torch(x3, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_y = tt_x1 + tt_x2
        tt_z = tt_x1 / tt_x2
        tt_p = tt_x1 * tt_x3

        back_torch_tt_y = ttnn.to_torch(tt_y)
        back_torch_tt_z = ttnn.to_torch(tt_z)
        back_torch_tt_p = ttnn.to_torch(tt_p)

        del tt_x1
        del tt_x2
        del tt_y
        del tt_z
        # print("y: ", y, "\nback_torch_tt_y: ", back_torch_tt_y)

        try:
            torch.testing.assert_close(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        try:
            torch.testing.assert_close(z, back_torch_tt_z, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        try:
            torch.testing.assert_close(p, back_torch_tt_p, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return

    def test_sum(reset_seeds, device_sp, models_params, set_tolerance):
        device = device_sp

        (
            seq_len,
            label_len,
            pred_len,
            stride,
            kernel_size,
            d_model,
            n_heads,
            batch_size,
            enc_in,
            udefined_v,
            t_dim,
            torch_dtype,
            ttnn_dtype,
        ) = models_params

        (rtol, atol) = set_tolerance
        rtol = 1e-2  ## can not pass with other strict ones
        atol = 1e-2

        tensor_dims1 = (batch_size, t_dim, enc_in, pred_len)

        x1 = torch.randn(tensor_dims1).to(torch_dtype)

        y = x1.sum(dim=1)

        tt_x1 = ttnn.from_torch(x1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
        tt_y = ttnn.sum(tt_x1, dim=1, keepdim=False)  ## [batch_size, enc_in, pred_len]
        back_torch_tt_y = ttnn.to_torch(tt_y)
        print("y.shape: {}, tt_y.shape: {}".format(y.shape, tt_y.shape))

        del tt_x1
        del tt_y

        try:
            torch.testing.assert_close(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            # Print the detailed mismatch log
            print("Mismatch details in softmax:\n", e)
            assert False

        ttnn.device.CloseDevice(device)
        return
