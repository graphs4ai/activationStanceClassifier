import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from typing import List, Union, Optional, Dict, Tuple


class Phi3MiniWrapper:
    """
    Wrapper for Microsoft Phi-3 Mini using TransformerLens.
    Supports activation extraction, neuron-level interventions,
    generation with steering, and soft stance scoring.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16, # bfloat16 is strongly recommended for Phi-3 Mini
        n_devices: int = 1,
    ):
        """
        Args:
            model_name: Phi-3 Mini checkpoint supported by TransformerLens
            device: "cuda" or "cpu"
            dtype: torch.bfloat16 is preferred over float16 for Phi-3 Mini
            n_devices: Number of GPUs to split the model across (model parallelism).
        """
        self.device = device
        self.n_devices = n_devices

        # Initialize the model via TransformerLens
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=dtype,
            n_devices=n_devices,
            trust_remote_code=True,
        )

        # Handle multi-GPU input routing
        self.input_device = "cuda:0" if n_devices > 1 else device

        # Ensure pad token exists (Phi often uses eos_token as pad_token if not set)
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        self.n_layers = self.model.cfg.n_layers

    def get_layer_activations(
        self,
        tokens: torch.Tensor,
        layers: Union[List[int], str] = [0],
        activation_multipliers: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Returns resid_pre activations for selected layers, concatenated
        along the feature dimension.
        
        activation_multipliers format: {"layer_2-neuron_1024": 1.5, ...}
        """

        if isinstance(layers, str):
            if layers.lower() == "all":
                layers = list(range(self.n_layers))
            else:
                raise ValueError("layers must be a list or 'all'")

        if not layers:
            raise ValueError("layers list cannot be empty")

        if activation_multipliers is None:
            activation_multipliers = {}

        # Parse neuron multipliers
        layer_neuron_multipliers: Dict[int, Dict[int, float]] = {}
        for name, mult in activation_multipliers.items():
            parts = name.split("-")
            layer = int(parts[0].split("_")[1])
            neuron = int(parts[1].split("_")[1])
            layer_neuron_multipliers.setdefault(layer, {})[neuron] = mult

        activations: Dict[int, torch.Tensor] = {}

        def make_layer_hook(layer_idx: int, neuron_mults: Optional[Dict[int, float]]):
            def hook(resid_pre: torch.Tensor, hook):
                if neuron_mults:
                    modified = resid_pre.clone()
                    for n, m in neuron_mults.items():
                        modified[:, :, n] *= m
                    activations[layer_idx] = modified.detach().cpu()
                    return modified
                else:
                    activations[layer_idx] = resid_pre.detach().cpu()
                    return resid_pre
            return hook

        intervention_layers = set(layer_neuron_multipliers.keys())
        all_hook_layers = set(layers) | intervention_layers

        fwd_hooks = []
        for layer in all_hook_layers:
            hook_point = f"blocks.{layer}.hook_resid_pre"
            fwd_hooks.append(
                (hook_point, make_layer_hook(layer, layer_neuron_multipliers.get(layer)))
            )

        stop_at_layer = max(all_hook_layers) + 1

        with torch.no_grad():
            self.model.run_with_hooks(
                tokens.to(self.input_device),
                fwd_hooks=fwd_hooks,
                stop_at_layer=stop_at_layer,
            )

        for layer in layers:
            if layer not in activations:
                raise RuntimeError(f"Missing activation for layer {layer}")

        return torch.cat([activations[l] for l in sorted(layers)], dim=-1)

    def generate_with_intervention(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        max_new_tokens: int = 10,
        temperature: Optional[float] = None,
        do_sample: bool = False,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        verbose: bool = False,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generates tokens while continuously applying multipliers to specific neurons.
        """

        if eos_token_id is None:
            eos_token_id = self.model.tokenizer.eos_token_id

        if not activation_multipliers:
            return self.model.generate(
                input_ids.to(self.input_device),
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 1.0,
                do_sample=do_sample,
                stop_at_eos=stop_at_eos,
                eos_token_id=eos_token_id,
                verbose=verbose,
                **generate_kwargs,
            )

        # Parse multipliers
        layer_neuron_multipliers: Dict[int, Dict[int, float]] = {}
        for name, mult in activation_multipliers.items():
            parts = name.split("-")
            layer = int(parts[0].split("_")[1])
            neuron = int(parts[1].split("_")[1])
            layer_neuron_multipliers.setdefault(layer, {})[neuron] = mult

        # Apply interventions only to the prompt tokens (buffer prevents interfering with new tokens)
        buffer_size = 3
        prompt_len = max(0, input_ids.shape[-1] - buffer_size)

        def make_hook(neuron_mults: Dict[int, float]):
            def hook(resid_pre: torch.Tensor, hook):
                modified = resid_pre.clone()
                for n, m in neuron_mults.items():
                    modified[:, :prompt_len, n] *= m
                return modified
            return hook

        hooks = []
        for layer, mults in layer_neuron_multipliers.items():
            hooks.append((f"blocks.{layer}.hook_resid_pre", make_hook(mults)))

        with torch.no_grad():
            for hp, fn in hooks:
                self.model.add_hook(hp, fn)
            try:
                out = self.model.generate(
                    input_ids.to(self.input_device),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature is not None else 1.0,
                    do_sample=do_sample,
                    stop_at_eos=stop_at_eos,
                    eos_token_id=eos_token_id,
                    verbose=verbose,
                    **generate_kwargs,
                )
            finally:
                self.model.reset_hooks()

        return out

    def get_stance_token_ids(self, language: str = "en") -> Tuple[int, int]:
        """
        Gets the token IDs for positive (Agree) and negative (Disagree) stance words.
        """
        if language == "pt":
            pos = "Con"
            neg = "Dis"
        else:
            pos = "Agree"
            neg = "Disagree"

        pos_id = self.model.tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = self.model.tokenizer.encode(neg, add_special_tokens=False)[0]
        
        return pos_id, neg_id

    def get_soft_stance_score(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        positive_token_id: Optional[int] = None,
        negative_token_id: Optional[int] = None,
        language: str = "en",
    ) -> Tuple[float, float]:
        """
        Calculates the stance by comparing the logits of positive and negative tokens.
        """

        if positive_token_id is None or negative_token_id is None:
            pos, neg = self.get_stance_token_ids(language)
            positive_token_id = positive_token_id or pos
            negative_token_id = negative_token_id or neg

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.input_device)

        if not activation_multipliers:
            with torch.no_grad():
                logits = self.model(input_ids)
        else:
            layer_neuron_multipliers: Dict[int, Dict[int, float]] = {}
            for name, mult in activation_multipliers.items():
                parts = name.split("-")
                layer = int(parts[0].split("_")[1])
                neuron = int(parts[1].split("_")[1])
                layer_neuron_multipliers.setdefault(layer, {})[neuron] = mult

            def make_hook(neuron_mults):
                def hook(resid_pre, hook):
                    modified = resid_pre.clone()
                    for n, m in neuron_mults.items():
                        modified[:, :, n] *= m
                    return modified
                return hook

            hooks = [
                (f"blocks.{l}.hook_resid_pre", make_hook(m))
                for l, m in layer_neuron_multipliers.items()
            ]

            with torch.no_grad():
                logits = self.model.run_with_hooks(input_ids, fwd_hooks=hooks)

        last_logits = logits[0, -1]
        probs = F.softmax(last_logits, dim=-1)

        p_pos = probs[positive_token_id].item()
        p_neg = probs[negative_token_id].item()

        return p_pos - p_neg, p_pos + p_neg
