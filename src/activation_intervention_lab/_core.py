"""Adapted TransformerLens serial core; bounded residual batching is opt-in.

Copyright (c) 2022 TransformerLensOrg. MIT; see vendor/transformerlens/LICENSE.
Source commit: bd55eab6737584685b921fdfcd15f191579e64b1.
The serial loop, metric validation, output axes and reductions remain copied.
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import pandas as pd
import torch
from jaxtyping import Int
from tqdm.auto import tqdm
import transformer_lens.utilities as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_protocol import TransformerLensModel
from transformer_lens.patching import (
    AxisNames, CorruptedActivation, PatchedActivation, MetricFn, PatchingResult,
    _REDUCE_MODES, make_df_from_ranges, layer_pos_patch_setter,
)
from .batching import validate_residual_sweep, batched_residual_runs

def generic_activation_patch(
    model: TransformerLensModel,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Union[MetricFn, Mapping[str, MetricFn]],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    *,
    reduce: str = "scalar",  # Reduce; widened so the ValueError below fires before beartype
    baseline: bool = False,
    patch_batch_size: int = 1,
    max_batch_tokens: int = 4096,
    max_batch_bytes: int = 128 * 1024 * 1024,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame], PatchingResult]:
    """
    A generic function to do activation patching, will be specialised to specific use cases.

    Activation patching is about studying the counterfactual effect of a specific activation between a clean run and a corrupted run. The idea is have two inputs, clean and corrupted, which have two different outputs, and differ in some key detail. Eg "The Eiffel Tower is in" vs "The Colosseum is in". Then to take a cached set of activations from the "clean" run, and a set of corrupted.

    Internally, the key function comes from three things: A list of tuples of indices (eg (layer, position, head_index)), a index_to_act_name function which identifies the right activation for each index, a patch_setter function which takes the corrupted activation, the index and the clean cache, and a metric for how well the patched model has recovered.

    The indices can either be given explicitly as a pandas dataframe, or by listing the relevant axis names and having them inferred from the tokens and the model config. It is assumed that the first column is always layer.

    This function then iterates over every tuple of indices, does the relevant patch, and stores it

    Every patched forward runs the whole corrupted batch, so per-example values are
    available for free: ``reduce="none"`` keeps them (the metric must then return a
    ``[batch]`` tensor) and returns a :class:`PatchingResult` with mean, standard
    error, bootstrap intervals and permutation p-values on top; ``reduce="mean"``
    averages a ``[batch]`` metric back to today's tensor; a mapping of metrics
    evaluates all of them on each patched forward. ``baseline=True`` adds one
    unpatched corrupted forward so effects and normalized recovery can be reported.

    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc), or a mapping of names to such functions. Must return a scalar under ``reduce="scalar"`` and a ``[batch]`` tensor otherwise.
        patch_setter: A function which acts on (corrupted_activation, index, clean_cache) to edit the activation and patch in the relevant chunk of the clean activation
        activation_name: The name of the activation being patched
        index_axis_names: The names of the axes to (fully) iterate over, implicitly fills in index_df
        index_df: The dataframe of indices, columns are axis names and each row is a tuple of indices. Will be inferred from index_axis_names if not given. When this is input, the output will be a flattened tensor with an element per row of index_df
        return_index_df: A Boolean flag for whether to return the dataframe of indices too (tensor return only; a ``PatchingResult`` always carries ``index_df``)
        reduce: ``"scalar"`` (default, metric returns a scalar, tensor return), ``"mean"`` (metric returns ``[batch]``, averaged, tensor return) or ``"none"`` (metric returns ``[batch]``, kept, ``PatchingResult`` return)
        baseline: Also evaluate the metric on the unpatched corrupted run and return a ``PatchingResult`` carrying it

    Returns:
        patched_output: The tensor of the patching metric for each patch. By default it has one dimension for each index dimension, via index_df set explicitly it is flattened with one element per row.
        index_df *optional*: The dataframe of indices
        Or a :class:`PatchingResult` when ``reduce="none"``, ``baseline=True`` or a metric mapping was given.
    """
    # Lazy import: tools.analysis imports this module's neighbours at package import.
    from transformer_lens.tools.analysis._model_state import require_eval_mode

    require_eval_mode(model, operation="activation patching")
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    metrics: Dict[str, MetricFn]
    if isinstance(patching_metric, Mapping):
        has_metric_axis = True
        metrics = dict(patching_metric)
        if not metrics:
            raise ValueError("patching_metric mapping must contain at least one metric")
    else:
        has_metric_axis = False
        metrics = {"metric": patching_metric}
    want_result = has_metric_axis or reduce == "none" or baseline
    if want_result and return_index_df:
        raise ValueError(
            "return_index_df only applies to the tensor return; a PatchingResult carries index_df"
        )

    if index_df is None:
        assert index_axis_names is not None
        number_of_heads = model.cfg.n_heads
        # For some models, the number of key value heads is not the same as the number of attention heads
        if activation_name in ["k", "v"] and model.cfg.n_key_value_heads is not None:
            number_of_heads = model.cfg.n_key_value_heads

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],
            "head_index": number_of_heads,
        }
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [max_axis_range[axis_name] for axis_name in index_axis_names]

        # Get the dataframe where each row is a tuple of indices
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()
        flattened_output = True

    validate_residual_sweep(model, corrupted_tokens, clean_cache, index_df, activation_name, patch_setter, patch_batch_size, max_batch_tokens, max_batch_bytes)
    batch_size = int(corrupted_tokens.shape[0])
    per_example = reduce != "scalar"
    cell_shape = (len(index_df),) if flattened_output else tuple(index_axis_max_range)
    store_shape = (len(metrics),) + cell_shape + ((batch_size,) if per_example else ())
    # Create an empty tensor to show the patched metric for each patch
    store = torch.zeros(store_shape, device=model.cfg.device)

    def evaluate(logits: torch.Tensor) -> torch.Tensor:
        """Every metric on one set of logits, shaped [metric, batch?] after validation."""
        rows = []
        for name, fn in metrics.items():
            out = torch.as_tensor(fn(logits))
            if per_example:
                if out.ndim != 1 or out.shape[0] != batch_size:
                    raise ValueError(
                        f"patching_metric {name!r} must return a [batch] tensor of length "
                        f"{batch_size} under reduce={reduce!r}, got shape {tuple(out.shape)}"
                    )
            elif out.ndim != 0:
                raise ValueError(
                    f"patching_metric {name!r} must return a scalar under reduce='scalar', got "
                    f"shape {tuple(out.shape)}; return a [batch] tensor and pass reduce='mean' "
                    "or reduce='none' to keep per-example values"
                )
            rows.append(out.detach())
        return torch.stack(rows).to(store.dtype)

    baseline_values: Optional[torch.Tensor] = None
    if baseline:
        # Same entry point as the patched runs, with no hooks attached.
        baseline_values = evaluate(model.run_with_hooks(corrupted_tokens, fwd_hooks=[]))

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        if corrupted_activation.requires_grad:
            corrupted_activation = corrupted_activation.clone()
        return patch_setter(corrupted_activation, index, clean_activation)

    if patch_batch_size == 1:
        for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
            index = index_row[1].to_list()

            # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
            current_activation_name = utils.get_act_name(activation_name, layer=index[0])

            # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
            current_hook = partial(
                patching_hook,
                index=index,
                clean_activation=clean_cache[current_activation_name],
            )

            patched_logits = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(current_activation_name, current_hook)]
            )
            values = evaluate(patched_logits)
            if flattened_output:
                store[:, c] = values
            else:
                store[(slice(None), *index)] = values

    else:
        for c, index, logits in batched_residual_runs(model, corrupted_tokens, clean_cache, index_df, patch_batch_size):
            values = evaluate(logits)
            if flattened_output:
                store[:, c] = values
            else:
                store[(slice(None), *index)] = values

    if reduce == "mean":
        store = store.mean(dim=-1)
        if baseline_values is not None:
            baseline_values = baseline_values.mean(dim=-1)

    if not want_result:
        patched_metric_output = store[0]
        if return_index_df:
            return patched_metric_output, index_df
        return patched_metric_output

    return PatchingResult(
        values=store if has_metric_axis else store[0],
        axis_names=[] if flattened_output else list(index_axis_names or []),
        index_df=index_df,
        metric_names=list(metrics),
        reduced=reduce != "none",
        baseline=None
        if baseline_values is None
        else (baseline_values if has_metric_axis else baseline_values[0]),
        has_metric_axis=has_metric_axis,
    )
