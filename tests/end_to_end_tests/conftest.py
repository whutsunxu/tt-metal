# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
import numpy as np

import ttnn


@pytest.fixture(scope="function")
def first_grayskull_device():
    device = ttnn.CreateDevice(0)
    yield device

    ttnn.CloseDevice(device)


"""must add the autouse=True with class level definition,
   otherwise the seed settings doen't work.
   (the reason is not clear yet)
"""


@pytest.fixture(scope="function", autouse=True)
def reset_seeds():
    print("call reset_seeds")
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield


@pytest.fixture(scope="function")
def models_params():
    seq_len = 336
    label_len = 336
    pred_len = 96
    stride = 1
    kernel_size = 25
    d_model = 512
    n_heads = 8
    batch_size = 8
    enc_in = 321
    udefined_v = 4
    t_dim = 1
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.float32

    yield seq_len, label_len, pred_len, stride, kernel_size, d_model, n_heads, batch_size, enc_in, udefined_v, t_dim, torch_dtype, ttnn_dtype


@pytest.fixture(scope="function")
def set_tolerance():
    rtol = 1e-5
    atol = 1e-5
    yield rtol, atol
