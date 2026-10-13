"""Bounded independent residual patches, following the upstream serial setter."""
from functools import partial
from numbers import Integral

import torch
from transformer_lens import utilities as utils
from transformer_lens.patching import layer_pos_patch_setter


def estimated_forward_bytes(model, tokens, patches):
    """Admission estimate for fp32 outputs, residual work and attention scores.

    This is a tensor-size estimate, not an upper bound on the process allocator.
    """
    batch, sequence = tokens.shape
    width, vocab, heads = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_heads
    item_size = torch.empty((), dtype=model.cfg.dtype).element_size()
    return patches * batch * (sequence * (vocab + 12 * width) + 2 * heads * sequence * sequence) * item_size


def validate_residual_sweep(model, tokens, cache, indices, activation_name, setter,
                            patches, max_tokens, max_bytes):
    for name, value in [("patch_batch_size", patches), ("max_batch_tokens", max_tokens), ("max_batch_bytes", max_bytes)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if activation_name != "resid_pre" or setter is not layer_pos_patch_setter:
        raise ValueError("this copied core supports resid_pre with layer_pos_patch_setter only")
    if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64) or min(tokens.shape) <= 0:
        raise ValueError("tokens must be a non-empty integer [example, position] tensor")
    if list(indices.columns) != ["layer", "pos"]:
        raise ValueError("index_df columns must be ['layer', 'pos'] in that order")
    admitted = min(patches, len(indices))
    if admitted * tokens.numel() > max_tokens:
        raise ValueError("patch batch exceeds max_batch_tokens")
    if estimated_forward_bytes(model, tokens, admitted) > max_bytes:
        raise ValueError("patch batch exceeds max_batch_bytes tensor estimate")
    checked = set()
    for row in indices.itertuples(index=False, name=None):
        layer, pos = row
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in row):
            raise ValueError("layer and position must be integer indices")
        if not 0 <= layer < model.cfg.n_layers or not 0 <= pos < tokens.shape[1]:
            raise ValueError(f"patch index outside model/input bounds: {row}")
        name = utils.get_act_name("resid_pre", layer=layer)
        if name not in checked:
            try:
                clean = cache[name]
            except KeyError as exc:
                raise ValueError(f"missing clean residual cache entry: {name}") from exc
            shape = (tokens.shape[0], tokens.shape[1], model.cfg.d_model)
            if not isinstance(clean, torch.Tensor) or tuple(clean.shape) != shape:
                raise ValueError(f"clean cache {name} must have shape {shape}")
            if not clean.is_floating_point() or clean.device != tokens.device:
                raise ValueError(f"clean cache {name} must be floating point on the input device")
            if clean.dtype != model.cfg.dtype:
                raise ValueError(f"clean cache {name} dtype must equal model dtype {model.cfg.dtype}")
            checked.add(name)


def batched_residual_runs(model, tokens, cache, indices, patches):
    """Each contiguous example slice carries one patch and independent forward state."""
    examples = tokens.shape[0]
    rows = list(indices.itertuples(index=False, name=None))
    for start in range(0, len(rows), patches):
        chunk = rows[start:start + patches]
        hooks_by_layer = {}
        for local, (layer, pos) in enumerate(chunk):
            hooks_by_layer.setdefault(layer, []).append((local, pos))

        def patch_hook(activation, hook, *, edits, clean):
            # Clone also protects the caller-owned clean cache from aliases.
            activation = activation.clone()
            for local, pos in edits:
                lo, hi = local * examples, (local + 1) * examples
                layer_pos_patch_setter(activation[lo:hi], [0, pos], clean)
            return activation

        hooks = []
        for layer, edits in hooks_by_layer.items():
            name = utils.get_act_name("resid_pre", layer=layer)
            hooks.append((name, partial(patch_hook, edits=edits, clean=cache[name])))
        expanded = tokens.repeat(len(chunk), 1)
        # Upstream context scopes remove only the hooks added here, including on error.
        with model.hooks(fwd_hooks=hooks):
            logits = model(expanded)
        for local, index in enumerate(chunk):
            yield start + local, index, logits[local * examples:(local + 1) * examples]
        del logits, expanded, hooks
