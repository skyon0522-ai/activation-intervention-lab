"""Local, reproducible residual-stream intervention experiments."""


def residual_patch_sweep(model, corrupted_tokens, clean_cache, metric, *, index_df=None,
                         patch_batch_size=1, reduce="none", baseline=True,
                         max_batch_tokens=4096, max_batch_bytes=128 * 1024 * 1024):
    """Preserve [metric?, layer, position, example?] or flat index-row axes."""
    from ._core import generic_activation_patch
    from transformer_lens.patching import layer_pos_patch_setter
    return generic_activation_patch(
        model, corrupted_tokens, clean_cache, metric, layer_pos_patch_setter,
        "resid_pre", index_axis_names=["layer", "pos"] if index_df is None else None,
        index_df=index_df, patch_batch_size=patch_batch_size, reduce=reduce,
        baseline=baseline, max_batch_tokens=max_batch_tokens, max_batch_bytes=max_batch_bytes,
    )


__all__ = ["residual_patch_sweep"]
