# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from .state_dict_utils import _compress_state_dict

# Mapping from data type to number of bits per scalar
DTYPE_SIZE_MAP = {
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.float16: 16,
    torch.float32: 32,
    torch.float64: 64,
    torch.long: 64,
}


def bits_to_mb(bits: int) -> float:
    """Convert from bits to mega bytes"""
    return bits / 8 / 1024 / 1024


def bits_to_kb(bits: int) -> float:
    return bits / 8 / 1024


def compute_model_nbits(model: torch.nn.Module) -> int:
    """Given a model, compressed or uncompressed, computes its size in bits.
    Compressed layers are assumed to use codebooks with half precision.

    Parameters:
        model: Model to compute the size of
    Returns:
        total_size_bits: Total size of the model in bits
    """

    compressed_dict = _compress_state_dict(model.state_dict(), float16_codebooks=True)

    total_size_bits = 0
    total_n_parameters = 0

    logging.debug("=== Computing model size ===")

    for module_name, module in compressed_dict.items():
        if "codes_matrix" in module_name:
            code_type = module.dtype
            n_bits_per_code = DTYPE_SIZE_MAP[code_type]

            n_codes = module.numel()
            code_size_bits = n_bits_per_code*n_codes
            total_size_bits += code_size_bits
            total_n_parameters += int(total_size_bits/32)
            continue

        # Batch norms, biases, and other elements can be counted according to their data type and number of parameters
        n_parameters = module.numel()
        total_n_parameters += n_parameters
        parameters_type = module.dtype
        parameters_size = DTYPE_SIZE_MAP[module.dtype]

        module_size_bits = n_parameters * parameters_size

        logging.debug(
            f"{module_name} of type {parameters_type} has {n_parameters} parameters and takes up {module_size_bits / 8} bytes"
        )

        total_size_bits += module_size_bits

        # if "codebook" in module_name:
        #     # Add the corresponding codes matrix. We do this here because we need to know the codebook size to estimate
        #     # the true minimum code size (independent of byte word length)
        #     codes_param = module_name.replace("codebook", "codes_matrix")
        #     if codes_param not in compressed_dict:
        #         continue
        #
        #     codes = compressed_dict[codes_param]
        #
        #     # This is safe since our codebooks are powers of 2
        #     n_bits_per_code = int(round(math.log(module.size(0), 2)))
        #     n_codes = codes.numel()
        #     module_size_bits = n_bits_per_code * n_codes
        #
        #     logging.debug(f"{codes_param} has {n_codes} codes and takes up {module_size_bits / 8} bytes")
        #
        #     total_size_bits += module_size_bits

    return total_size_bits


def log_compression_ratio(
    uncompressed_model_size_bits: int, compressed_model_size_bits: int, summary_writer: Optional[SummaryWriter] = None
) -> None:
    """Compute stats about model compression and log them to both the standard logger and tensorboard

    Parameters:
        uncompressed_model_size_bits: The size of the uncompressed model in bits
        compressed_model_size_bits: The size of the compressed model in bits
        summary_writer: Tensorbard logger to writer compression params etc. Optional. Not passing it disables logging
    """
    model_size_log = "\n" + "\n".join(
        [
            f"uncompressed (bits):   {uncompressed_model_size_bits}",
            f"compressed (bits):     {compressed_model_size_bits}",
            f"uncompressed (MB/KB):  {bits_to_mb(uncompressed_model_size_bits):.2f}/{bits_to_kb(uncompressed_model_size_bits):.2f}",
            f"compressed (MB/KB):    {bits_to_mb(compressed_model_size_bits):.2f}/{bits_to_kb(compressed_model_size_bits):.2f}",
            f"compression ratio:     {(uncompressed_model_size_bits / compressed_model_size_bits):.2f}",
        ]
    )
    logging.info(model_size_log)

    if summary_writer is not None:
        summary_writer.add_text("model", model_size_log)