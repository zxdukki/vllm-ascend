#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/test_offline_inference.py`.
"""
import os

import vllm  # noqa: F401

from tests.conftest import VllmRunner

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_DBO"] = "1"


def test_deepseek_model_with_dbo():
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 100
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "deepseek-ai/DeepSeek-V2-Lite",
            dtype=dtype,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
