"""
Model Factory for selecting between different model wrappers.

This module provides a unified interface for creating model wrappers
based on configuration, allowing easy switching between supported model families.
"""

from typing import Union, Optional
from omegaconf import DictConfig


def get_model_wrapper(cfg: DictConfig, device: str = "auto"):
    """
    Factory function to create the appropriate model wrapper based on config.

    Args:
        cfg: Hydra/OmegaConf config containing model settings.
             Expected structure:
               model:
                                 name: "meta-llama/Llama-3.1-8B-Instruct"  # or another supported model name
                                wrapper: "llama"  # "llama", "gemma", "qwen", "phi", or "mistral"
        device: Override device from config. If "auto", uses cfg.extraction.device.

    Returns:
        Model wrapper instance for the selected wrapper type.

    Raises:
        ValueError: If wrapper type is not recognized
    """
    # Determine device
    import torch
    if device == "auto":
        device = cfg.get("extraction", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Get model config with defaults for backward compatibility
    model_cfg = cfg.get("model", {})
    wrapper_type = model_cfg.get("wrapper", "llama").lower()
    model_name = model_cfg.get("name", None)
    n_devices = model_cfg.get("n_devices", 1)

    # Auto-cap n_devices at the number of available GPUs
    if device != "cpu" and n_devices > 1:
        gpu_count = torch.cuda.device_count()
        if gpu_count < n_devices:
            print(
                f"Warning: n_devices={n_devices} requested but only {gpu_count} GPU(s) available. "
                f"Using n_devices={gpu_count}."
            )
            n_devices = max(gpu_count, 1)

    # Resolve dtype from config string
    dtype_str = model_cfg.get("dtype", "float16").lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown dtype: '{dtype_str}'. "
            f"Supported: {list(dtype_map.keys())}"
        )
    dtype = dtype_map[dtype_str]

    if wrapper_type == "llama":
        from llama_3dot1_wrapper import Llama3dot1Wrapper
        if model_name:
            model_wrapper = Llama3dot1Wrapper(
                model_name=model_name, device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    f"Failed to initialize tokenizer for model: {model_name}")
            return model_wrapper
        else:
            model_wrapper = Llama3dot1Wrapper(
                device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    "Failed to initialize tokenizer for default Llama model")
            return model_wrapper

    elif wrapper_type == "gemma":
        from gemma_3_wrapper import Gemma3Wrapper
        if model_name:
            model_wrapper = Gemma3Wrapper(
                model_name=model_name, device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    f"Failed to initialize tokenizer for model: {model_name}")
            return model_wrapper
        else:
            model_wrapper = Gemma3Wrapper(
                device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    "Failed to initialize tokenizer for default Gemma model")
            return model_wrapper

    elif wrapper_type == "qwen":
        from qwen_3_wrapper import Qwen3Wrapper
        if model_name:
            model_wrapper = Qwen3Wrapper(
                model_name=model_name, device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    f"Failed to initialize tokenizer for model: {model_name}")
            return model_wrapper
        else:
            model_wrapper = Qwen3Wrapper(
                device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    "Failed to initialize tokenizer for default Qwen model")
            return model_wrapper

    elif wrapper_type == "phi":
        from phi_3_mini_wrapper import Phi3MiniWrapper
        if model_name:
            model_wrapper = Phi3MiniWrapper(
                model_name=model_name, device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    f"Failed to initialize tokenizer for model: {model_name}")
            return model_wrapper
        else:
            model_wrapper = Phi3MiniWrapper(
                device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    "Failed to initialize tokenizer for default Phi model")
            return model_wrapper

    elif wrapper_type == "mistral":
        from mistral_7b_wrapper import Mistral7BWrapper
        if model_name:
            model_wrapper = Mistral7BWrapper(
                model_name=model_name, device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    f"Failed to initialize tokenizer for model: {model_name}")
            return model_wrapper
        else:
            model_wrapper = Mistral7BWrapper(
                device=device, dtype=dtype, n_devices=n_devices)
            if model_wrapper.model.tokenizer is None:
                raise ValueError(
                    "Failed to initialize tokenizer for default Mistral model")
            return model_wrapper

    else:
        raise ValueError(
            f"Unknown wrapper type: '{wrapper_type}'. "
            f"Supported types: 'llama', 'gemma', 'qwen', 'phi', 'mistral'"
        )


def get_wrapper_class(wrapper_type: str):
    """
    Get the wrapper class without instantiating it.

    Args:
        wrapper_type: "llama", "gemma", "qwen", "phi", or "mistral"

    Returns:
        The wrapper class (not an instance)
    """
    wrapper_type = wrapper_type.lower()

    if wrapper_type == "llama":
        from llama_3dot1_wrapper import Llama3dot1Wrapper
        return Llama3dot1Wrapper
    elif wrapper_type == "gemma":
        from gemma_3_wrapper import Gemma3Wrapper
        return Gemma3Wrapper
    elif wrapper_type == "qwen":
        from qwen_3_wrapper import Qwen3Wrapper
        return Qwen3Wrapper
    elif wrapper_type == "phi":
        from phi_3_mini_wrapper import Phi3MiniWrapper
        return Phi3MiniWrapper
    elif wrapper_type == "mistral":
        from mistral_7b_wrapper import Mistral7BWrapper
        return Mistral7BWrapper
    else:
        raise ValueError(
            f"Unknown wrapper type: '{wrapper_type}'. "
            f"Supported types: 'llama', 'gemma', 'qwen', 'phi', 'mistral'"
        )
