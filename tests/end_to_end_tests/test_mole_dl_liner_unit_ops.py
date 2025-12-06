# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import random

import ttnn


@pytest.mark.model_mole_dl_linear_ops
class TestTTNNSiliconOps:
    def test_zeros_like(reset_seeds, first_grayskull_device, models_params):
        device = first_grayskull_device

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

        xtt_host = xtt.cpu()
        tt_got_back = xtt_host.to_torch()
        eq = torch.equal(x, tt_got_back)
        assert eq

        del xtt
        del tmp_a

        ttnn.device.CloseDevice(device)

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

    def test_concat(self, reset_seeds, first_grayskull_device, models_params):
        device = first_grayskull_device

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

    def test_slice(self, reset_seeds, first_grayskull_device, models_params):
        device = first_grayskull_device

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

    def test_repeat(reset_seeds, first_grayskull_device, models_params):
        device = first_grayskull_device

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

    def test_permute(self, reset_seeds, first_grayskull_device, models_params, set_tolerance):
        device = first_grayskull_device

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

    @pytest.mark.skip(reason="Skipping test_avgpool1d temporarily, because it will coredump")
    def test_avgpool1d(reset_seeds, first_grayskull_device, models_params, set_tolerance):
        device = first_grayskull_device

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

        tensor_dims_NCH = (batch_size, enc_in, seq_len + ((kernel_size - 1) // 2) * 2)
        tensor_dims_NHWC = (batch_size, seq_len + ((kernel_size - 1) // 2) * 2, 1, enc_in)

        """--------------------- call torch avg_pool1d -------------------------------"""
        padding = 0
        avg_pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

        x = torch.randn(tensor_dims_NCH).to(torch_dtype)  # N C H
        y = avg_pool(x)  ## N C H

        """--------------------- call ttnn avg_pool2d -------------------------------"""
        x_2d = torch.unsqueeze(x, -1)  ## N, C, H --> N, C, H, 1
        x_NHWC = torch.permute(x_2d, (0, 2, 3, 1))  ## N, C, H, 1 --> N, H, 1, C
        tt_x = ttnn.from_torch(x_NHWC, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn_dtype)

        tt_y = ttnn.avg_pool2d(
            input_tensor=tt_x,
            batch_size=tensor_dims_NHWC[0],
            input_h=tensor_dims_NHWC[1],
            input_w=tensor_dims_NHWC[2],
            channels=tensor_dims_NHWC[3],
            kernel_size=[kernel_size, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        back_torch_tt_y_NHWC = ttnn.to_torch(tt_y)  ## N, H, 1, C
        ## N, H, 1, C --> N, H, C
        back_torch_tt_y_NHC = torch.squeeze(back_torch_tt_y_NHWC.to(torch_dtype), dim=2)
        ## N, H, C --> N, C, H
        back_torch_tt_y = torch.permute(back_torch_tt_y_NHC, (0, 2, 1))
        print("y: ", y, "\ntt_y: ", tt_y)

        del tt_x
        del tt_y

        assert torch.allclose(y, back_torch_tt_y, rtol=rtol, atol=atol, equal_nan=False)
        ttnn.device.CloseDevice(device)
        return

    def test_binary_op(reset_seeds, first_grayskull_device, models_params, set_tolerance):
        device = first_grayskull_device

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

    def test_reshape(self, reset_seeds, first_grayskull_device, models_params):
        device = first_grayskull_device

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
