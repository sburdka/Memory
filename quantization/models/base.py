"""
BasePatch  — model-agnostic layer replacement machinery.
BaseMiLoModel — compress / save / load lifecycle.
"""

from __future__ import annotations

import gc
import json
import os
from abc import abstractmethod
from functools import partial
from os.path import join as pjoin
from typing import Callable, Union

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from tqdm import tqdm

from ..quantize import MiLoLinear, BaseCompressConfig
from ..compensator import rank_generate, load_compensators

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUANT_LAYERS = [nn.Linear, MiLoLinear]
_IGNORE_LINEAR = ["lm_head"]


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_leaf(module: nn.Module) -> bool:
    return len(module._modules) == 0


def _find_parent(model: nn.Module, name: str) -> nn.Module:
    parent = model
    for part in name.split(".")[:-1]:
        parent = parent._modules[part]
    return parent


def _name_to_linear_tag(name: str) -> str:
    return ".".join(
        p for p in name.split(".")
        if p not in ("model", "layers") and not p.isnumeric()
    )


def _get_all_children(model: nn.Module, ignore: list = []) -> list[str]:
    return [
        name for name, mod in model.named_modules()
        if _is_leaf(mod) and name.split(".")[-1] not in ignore
    ]


def _get_linear_tags(model: nn.Module, ignore: list = _IGNORE_LINEAR) -> list[str]:
    tags: set[str] = set()
    for name, mod in model.named_modules():
        if type(mod) in _QUANT_LAYERS and name.split(".")[-1] not in ignore:
            tags.add(_name_to_linear_tag(name))
    return list(tags)


def _forward_device_hooked(self, *args, **kwargs):
    """Hook that moves inputs to the module's device before the forward call."""
    args = [a.to(self.device) if isinstance(a, torch.Tensor) else a for a in args]
    kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return self.forward_orig(*args, **kwargs)


# ---------------------------------------------------------------------------
# BasePatch
# ---------------------------------------------------------------------------

class BasePatch:
    """
    Defines how to walk the model graph and replace layers.
    Subclasses override get_linear_tags(), patch_nonlinearlayers(),
    and patch_linearlayers() to handle architecture-specific traversal.
    """

    @classmethod
    def get_linear_tags(cls) -> list[str]:
        return []

    @classmethod
    def get_ignore_layers(cls, model: nn.Module) -> list[str]:
        return [name for name, mod in model.named_modules() if not _is_leaf(mod)] + [""]

    @classmethod
    def set_auto_linear_tags(cls, model: nn.Module, ignore: list = _IGNORE_LINEAR) -> None:
        if not hasattr(model, "linear_tags"):
            tags = cls.get_linear_tags()
            model.linear_tags = tags if tags else _get_linear_tags(model, ignore)
            model.base_class = cls

    @classmethod
    def autoname_modules(cls, model: nn.Module) -> None:
        for name, mod in model.named_modules():
            mod.name = name

    @classmethod
    def freeze_model(cls, model: nn.Module) -> None:
        for p in model.parameters():
            p.requires_grad = False

    @classmethod
    def patch_nonlinearlayers(cls, model: nn.Module, patch_fct: Callable, verbose: bool = True) -> None:
        ignore = cls.get_ignore_layers(model)
        mapping = {
            name: mod for name, mod in model.named_modules()
            if type(mod) not in _QUANT_LAYERS and name not in ignore
        }
        for name in tqdm(mapping, disable=not verbose, desc="Patching non-linear"):
            setattr(_find_parent(model, name), name.split(".")[-1], patch_fct(mapping[name]))
        _cleanup()

    @classmethod
    def patch_linearlayers(
        cls, model: nn.Module, patch_fct: Callable, patch_params: dict, verbose: bool = True
    ) -> None:
        ignore = cls.get_ignore_layers(model)
        mapping = {
            name: mod for name, mod in model.named_modules()
            if type(mod) in _QUANT_LAYERS and name not in ignore
        }
        for name in tqdm(mapping, disable=not verbose, desc="Patching linear"):
            tag = _name_to_linear_tag(name)
            param = patch_params.get(tag)
            setattr(_find_parent(model, name), name.split(".")[-1], patch_fct(mapping[name], param))
        _cleanup()

    @classmethod
    def patch_model(
        cls,
        model: nn.Module,
        patch_nonlinear_fct: Callable,
        patch_linear_fct: Callable,
        patch_params: dict,
        verbose: bool = True,
    ) -> None:
        model.eval()
        cls.freeze_model(model)
        cls.autoname_modules(model)
        cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
        cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
        _cleanup()


# ---------------------------------------------------------------------------
# BaseMiLoModel
# ---------------------------------------------------------------------------

class BaseMiLoModel:
    """Compress / save / load lifecycle for MiLo quantised models."""

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def create_model(cls, save_dir: str, kwargs: dict):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def cache_model(cls, model: nn.Module, save_dir: str):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # File layout
    # ------------------------------------------------------------------

    @classmethod
    def _weight_path(cls, save_dir: str) -> str:
        return pjoin(save_dir, "qmodel.pt")

    @classmethod
    def _config_path(cls, save_dir: str) -> str:
        return pjoin(save_dir, "config.json")

    @classmethod
    def _compensator_path(cls, save_dir: str) -> str:
        return pjoin(save_dir, "compensators.pt")

    @classmethod
    def _ranks_path(cls, save_dir: str) -> str:
        return pjoin(save_dir, "ranks.json")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @classmethod
    def setup_model(cls, model: nn.Module) -> None:
        cls.autoname_modules(model)
        cls.set_auto_linear_tags(model)

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    @classmethod
    def compress_model(
        cls,
        model: nn.Module,
        compress_config: BaseCompressConfig,
        compute_dtype: torch.dtype = torch.float16,
        device: Union[str, list, dict] = "cuda",
    ) -> nn.Module:
        cls.setup_model(model)

        # Assign ranks for all layers
        comp_params = compress_config["compensator_params"]
        comp_params["ranks"] = rank_generate(
            model.config._name_or_path,
            comp_params["sparse_rank"],
            comp_params["dense_rank"],
            comp_params.get("rank_strategy"),
        )

        # Build per-tag patch params
        if any(key in model.linear_tags for key in compress_config):
            patch_params = {tag: None for tag in model.linear_tags}
            patch_params.update(compress_config)
        else:
            patch_params = {tag: compress_config for tag in model.linear_tags}

        all_nodes = _get_all_children(model)
        try:
            core = model.model if hasattr(model, "model") else model
            n_blocks = len(core.layers)
            all_blocks = [f"model.layers.{i}" for i in range(n_blocks)]
        except Exception:
            all_blocks = []

        # Build device map
        if isinstance(device, str):
            device_map = {k: device for k in all_blocks + all_nodes}
            n_devices = 1
        elif isinstance(device, list):
            n_devices = len(device)
            device_map = {}
            step = max(len(all_blocks) // n_devices, 1)
            for k, node in enumerate(all_nodes):
                if ".layers" not in node:
                    device_map[node] = device[0]
            for k, node in enumerate(reversed(all_nodes)):
                if ".layers" not in node:
                    device_map[node] = device[-1]
            for i, blk in enumerate(all_blocks):
                device_map[blk] = device[min(i // step, n_devices - 1)]
        else:  # dict
            device_map = device
            n_devices = len(set(device_map.values()))
            all_blocks = list(device_map.keys())

        # Map all leaf nodes to their block's device
        node_to_block = {}
        for node in all_nodes:
            matches = [blk for blk in all_blocks if blk in node]
            node_to_block[node] = matches[-1] if matches else node
        for node in all_nodes:
            if node not in device_map:
                device_map[node] = device_map.get(node_to_block[node], "cpu")

        def _patch_linear(layer: nn.Linear, cfg):
            if isinstance(layer, MiLoLinear):
                return layer
            if cfg is None:
                return layer.to(device=device_map.get(layer.name, "cpu"), dtype=compute_dtype)
            out = MiLoLinear(layer, cfg, compute_dtype=compute_dtype, device=device_map.get(layer.name, "cpu"))
            out.device = device_map.get(layer.name, "cpu")
            return out

        def _patch_other(layer):
            dev = device_map.get(layer.name, "cpu")
            layer.device = dev
            return layer.to(device=dev, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)

        # Multi-device: insert device-switching hooks
        if n_devices > 1:
            core = model if hasattr(model, "layers") else model.model
            first_node = all_nodes[0]
            first_child = getattr(core, first_node.split(".")[-1])
            first_child.device = device_map.get(first_node, "cpu")
            first_child.forward_orig = first_child.forward
            first_child.forward = partial(_forward_device_hooked, first_child)
            setattr(core, first_node.split(".")[-1], first_child)

            for i, blk_layer in enumerate(core.layers):
                blk_name = f"model.layers.{i}"
                blk_layer.device = device_map.get(blk_name, "cpu")
                blk_layer.forward_orig = blk_layer.forward
                blk_layer.forward = partial(_forward_device_hooked, blk_layer)

        model.base_class = cls
        model.milo_compressed = True
        return model

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def serialize_weights(cls, model: nn.Module) -> dict:
        weights = {}
        ignore = cls.get_ignore_layers(model)
        for name, mod in model.named_modules():
            if name in ignore:
                continue
            try:
                mod.encoded_state_dict = False
                sd = mod.state_dict()
                if sd:
                    weights[name] = dict(sd)
            except Exception:
                pass
        return weights

    @classmethod
    def save_compressed(cls, model: nn.Module, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        cls.cache_model(model, save_dir)

        # Compensators
        comp = {name: mod.UV_quantized for name, mod in model.named_modules() if isinstance(mod, MiLoLinear)}
        torch.save(comp, cls._compensator_path(save_dir))

        # Ranks
        ranks = {}
        for _, mod in model.named_modules():
            if isinstance(mod, MiLoLinear) and mod.compress_config:
                ranks = mod.compress_config.get("compensator_params", {}).get("ranks", {})
                break
        with open(cls._ranks_path(save_dir), "w") as f:
            json.dump(ranks, f, indent=2)

        # Weights
        torch.save(cls.serialize_weights(model), cls._weight_path(save_dir))
        print(f"Saved compressed model to {save_dir}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def from_compressed(
        cls,
        save_dir_or_hub: str,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        cache_dir: Union[str, None] = "",
        **kwargs,
    ) -> nn.Module:
        # Resolve directory
        save_dir = save_dir_or_hub
        if cache_dir is not None:
            save_dir = pjoin(cache_dir, save_dir_or_hub)
        if not os.path.exists(cls._weight_path(save_dir)):
            save_dir = snapshot_download(repo_id=save_dir_or_hub, cache_dir=cache_dir or None)

        model = cls.create_model(save_dir, kwargs)
        model.save_dir = save_dir
        cls.setup_model(model)

        weights = torch.load(cls._weight_path(save_dir), map_location="cpu")
        all_compensators = torch.load(cls._compensator_path(save_dir), map_location="cpu")

        @torch.no_grad()
        def _load_module(module, params=None):
            if module.name not in weights:
                return module.to(device=device, dtype=compute_dtype)
            sd = weights[module.name]
            if "W_q" in sd:
                m = MiLoLinear(None, None, compute_dtype=compute_dtype, device=device)
                m.load_state_dict(sd)
                m.name = module.name
                return m
            for key, val in sd.items():
                setattr(module, key, nn.Parameter(val.to(device=device, dtype=compute_dtype), requires_grad=False))
            return module

        cls.patch_model(model, _load_module, _load_module, {t: None for t in model.linear_tags})

        with open(cls._ranks_path(save_dir)) as f:
            ranks = json.load(f)

        load_compensators(model, all_compensators, ranks)
        model.milo_compressed = True
        model.base_class = cls
        return model
