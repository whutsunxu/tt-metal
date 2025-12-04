# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
import numpy as np

import ttnn
from tests.scripts.common import run_process_and_get_result
from tests.scripts.common import get_updated_device_params


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


# TODO: Remove this when TG clusters are deprecated.
def is_tg_cluster():
    import ttnn

    return ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG


def first_available_tg_device():
    assert is_tg_cluster()
    # The id of the first user exposed device for a TG cluster is 4
    return 4


@pytest.fixture(scope="function")
def device_sp(request):
    import ttnn

    default_device_params = {"l1_small_size": 24576}
    override_params = getattr(request, "param", {})
    device_params = {**default_device_params, **override_params}

    device_id = 0  ##request.config.getoption("device_id")

    request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]

    # When initializing a single device on a TG system, we want to
    # target the first user exposed device, not device 0 (one of the
    # 4 gateway devices)
    if is_tg_cluster() and not device_id:
        device_id = first_available_tg_device()

    updated_device_params = get_updated_device_params(device_params)
    device = ttnn.CreateDevice(device_id=device_id, **updated_device_params)
    ttnn.SetDefaultDevice(device)

    yield device

    ttnn.close_device(device)
