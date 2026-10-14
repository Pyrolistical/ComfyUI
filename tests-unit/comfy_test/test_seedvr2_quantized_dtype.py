import unittest
from unittest.mock import patch

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management
from comfy.supported_models import SeedVR2


class TestSeedVR2QuantizedComputeDtype(unittest.TestCase):
    def test_fp16_manual_cast_is_promoted_to_bf16(self):
        config = SeedVR2({"image_model": "seedvr2"})

        with patch.object(comfy.model_management, "should_use_bf16", return_value=True):
            config.set_inference_dtype(torch.float8_e4m3fn, torch.float16)

        self.assertIs(config.manual_cast_dtype, torch.bfloat16)
