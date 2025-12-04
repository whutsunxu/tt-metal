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
        tt_z = ttnn.concat([tt_x, tt_y], dim=1)

        back_tt_z = tt_z.cpu()
        back_torch_tt_z = back_tt_z.to_torch().to(torch_dtype)

        # print("x: ", x, "\ny: ", y, "\nz: ", z)
        # print("tt_x: ", tt_x, "\ntt_y: ", tt_y, "\ntt_z: ", tt_z)

        eq = torch.equal(z, back_torch_tt_z)

        del tt_x
        del tt_y
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

        ttnn.device.CloseDevice(device)
