import pandas as pd
import pytest
import torch
import hashlib
import json
from pathlib import Path
from transformer_lens import patching as upstream

from activation_intervention_lab import residual_patch_sweep
from activation_intervention_lab.batching import estimated_forward_bytes
from activation_intervention_lab.model import load_model, clean_residual_cache, hook_count


@pytest.mark.parametrize("patches", [0, -1, True, 1.5])
def test_cli_experiment_rejects_invalid_batch_before_model_load(monkeypatch, patches):
    from activation_intervention_lab import experiment as runner
    def unexpected_load(*args, **kwargs):
        raise AssertionError("invalid batch must be rejected before loading a model")
    monkeypatch.setattr(runner, "load_model", unexpected_load)
    with pytest.raises(ValueError, match="positive integer"):
        runner.run_experiment(patch_batch_size=patches)


def test_cli_clamps_then_admits_before_batch_control_allocation(experiment, monkeypatch):
    from activation_intervention_lab import batching
    from activation_intervention_lab import experiment as runner
    model, clean, _, _, _, _ = experiment
    monkeypatch.setattr(runner, "load_model", lambda *args, **kwargs: model)
    calls = []
    forward = model.forward
    def counted(tokens, *args, **kwargs):
        calls.append(tokens.shape[0])
        return forward(tokens, *args, **kwargs)
    def oversized_estimate(current_model, tokens, patches):
        assert patches == current_model.cfg.n_layers * tokens.shape[1]
        return 128 * 1024 * 1024 + 1
    monkeypatch.setattr(model, "forward", counted)
    monkeypatch.setattr(runner, "estimated_forward_bytes", oversized_estimate)
    monkeypatch.setattr(batching, "estimated_forward_bytes", oversized_estimate)
    with pytest.raises(ValueError, match="max_batch_bytes") as rejected:
        runner.run_experiment(patch_batch_size=1_000_000_000, repeats=1)
    # Only the clean cache was built; no corrupt/batch-only forward was admitted.
    assert calls == [clean.shape[0]]
    assert rejected.value.partial_report["cache_released"] is True
    assert hook_count(model) == 0


@pytest.fixture(scope="module")
def experiment():
    model = load_model()
    clean = model.to_tokens(["The capital of France is", "The capital of Japan is"], prepend_bos=False)
    corrupt = model.to_tokens(["The capital of Italy is", "The capital of China is"], prepend_bos=False)
    with torch.inference_mode():
        _, cache = clean_residual_cache(model, clean)
    indices = pd.DataFrame([(0, 3), (3, 4), (1, 0), (0, 3), (4, 2), (5, 4), (2, 3)], columns=["layer", "pos"])
    answers = torch.tensor([model.tokenizer.encode(" Paris")[0], model.tokenizer.encode(" Tokyo")[0]])
    rivals = torch.tensor([model.tokenizer.encode(" Rome")[0], model.tokenizer.encode(" Beijing")[0]])
    def difference(logits):
        last = logits[:, -1]
        return last.gather(1, answers[:, None]).squeeze(1) - last.gather(1, rivals[:, None]).squeeze(1)
    def probability(logits):
        return logits[:, -1].softmax(-1).gather(1, answers[:, None]).squeeze(1)
    yield model, clean, corrupt, cache, indices, {"difference": difference, "probability": probability}
    cache.cache_dict.clear()
    assert hook_count(model) == 0


@pytest.mark.parametrize("patches", [1, 2, 4, 7])
@pytest.mark.parametrize("reduce", ["none", "mean", "scalar"])
def test_original_copied_and_batched_axes_and_reductions(experiment, patches, reduce):
    model, _, tokens, cache, indices, metrics = experiment
    if reduce == "scalar":
        metrics = {name: (lambda logits, fn=fn: fn(logits).mean()) for name, fn in metrics.items()}
    with torch.inference_mode():
        original = upstream.generic_activation_patch(
            model, tokens, cache, metrics, upstream.layer_pos_patch_setter, "resid_pre",
            index_df=indices, reduce=reduce, baseline=True,
        )
        actual = residual_patch_sweep(model, tokens, cache, metrics, index_df=indices,
                                      patch_batch_size=patches, reduce=reduce, baseline=True)
    torch.testing.assert_close(actual.values, original.values, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual.baseline, original.baseline, atol=0, rtol=0)
    assert actual.metric_names == list(metrics)
    assert actual.reduced == (reduce != "none")
    assert actual.axis_names == []
    assert actual.index_df.equals(indices)
    assert hook_count(model) == 0


def test_grid_axes_repeatability_and_cache_immutability(experiment):
    model, _, tokens, cache, _, metrics = experiment
    before = {key: value.clone() for key, value in cache.items()}
    with torch.inference_mode():
        first = residual_patch_sweep(model, tokens, cache, metrics, patch_batch_size=4)
        second = residual_patch_sweep(model, tokens, cache, metrics, patch_batch_size=4)
        serial = upstream.generic_activation_patch(
            model, tokens, cache, metrics, upstream.layer_pos_patch_setter, "resid_pre",
            index_axis_names=["layer", "pos"], reduce="none", baseline=True,
        )
    assert first.values.shape == (2, model.cfg.n_layers, tokens.shape[1], tokens.shape[0])
    assert first.axis_names == ["layer", "pos"]
    torch.testing.assert_close(first.values, second.values, atol=0, rtol=0)
    torch.testing.assert_close(first.values, serial.values, atol=1e-5, rtol=1e-5)
    for key, value in cache.items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_no_intervention_and_shared_prefix_control(experiment):
    model, _, tokens, cache, _, metrics = experiment
    indices = pd.DataFrame([(layer, 0) for layer in range(model.cfg.n_layers)], columns=["layer", "pos"])
    with torch.inference_mode():
        direct = model(tokens)
        unpatched = model.run_with_hooks(tokens, fwd_hooks=[])
        control = residual_patch_sweep(model, tokens, cache, metrics, index_df=indices, patch_batch_size=4)
        after = model(tokens)
    torch.testing.assert_close(direct, unpatched, atol=0, rtol=0)
    torch.testing.assert_close(direct, after, atol=0, rtol=0)
    torch.testing.assert_close(control.effect(), torch.zeros_like(control.values), atol=1e-5, rtol=0)


@pytest.mark.parametrize("row", [(-1, 0), (0, -1), (6, 0), (0, 5), (0.5, 1), (True, 1)])
def test_invalid_indices(experiment, row):
    model, _, tokens, cache, _, metrics = experiment
    with pytest.raises(ValueError):
        residual_patch_sweep(model, tokens, cache, metrics,
                             index_df=pd.DataFrame([row], columns=["layer", "pos"]), patch_batch_size=4)
    assert hook_count(model) == 0


@pytest.mark.parametrize("fault", ["shape", "missing", "dtype", "column_order", "rank", "budget_tokens", "budget_bytes", "batch_zero"])
def test_validation_before_hooks(experiment, fault):
    model, _, tokens, cache, indices, metrics = experiment
    modified = dict(cache.items())
    kwargs = {"index_df": indices, "patch_batch_size": 4}
    if fault == "shape":
        modified["blocks.0.hook_resid_pre"] = modified["blocks.0.hook_resid_pre"][:1]
    elif fault == "missing":
        del modified["blocks.0.hook_resid_pre"]
    elif fault == "dtype":
        modified["blocks.0.hook_resid_pre"] = modified["blocks.0.hook_resid_pre"].double()
    elif fault == "column_order":
        kwargs["index_df"] = indices[["pos", "layer"]]
    elif fault == "rank":
        tokens = tokens[0]
    elif fault == "budget_tokens":
        kwargs["max_batch_tokens"] = 1
    elif fault == "budget_bytes":
        kwargs["max_batch_bytes"] = 1
    elif fault == "batch_zero":
        kwargs["patch_batch_size"] = 0
    with pytest.raises(ValueError):
        residual_patch_sweep(model, tokens, modified, metrics, **kwargs)
    assert hook_count(model) == 0


def test_hook_exception_cleanup_preserves_caller_hook(experiment):
    model, _, tokens, cache, indices, metrics = experiment
    def caller_hook(value, hook):
        return value
    def fail(value, hook):
        raise RuntimeError("injected forward failure")
    with model.hooks(fwd_hooks=[("blocks.0.hook_in", caller_hook)]):
        caller_count = hook_count(model)
        with model.hooks(fwd_hooks=[("blocks.2.hook_in", fail)]):
            count = hook_count(model)
            with pytest.raises(RuntimeError, match="injected forward"):
                with torch.inference_mode():
                    residual_patch_sweep(model, tokens, cache, metrics, index_df=indices,
                                         patch_batch_size=4, baseline=False)
            assert hook_count(model) == count
        assert hook_count(model) == caller_count
    assert hook_count(model) == 0


def test_metric_failure_cleanup(experiment):
    model, _, tokens, cache, indices, _ = experiment
    def wrong_metric(logits):
        return logits.sum()
    with pytest.raises(ValueError, match="must return a"):
        with torch.inference_mode():
            residual_patch_sweep(model, tokens, cache, wrong_metric, index_df=indices,
                                 patch_batch_size=4, baseline=False)
    assert hook_count(model) == 0


def test_estimate_and_flat_empty(experiment):
    model, _, tokens, cache, _, metrics = experiment
    assert estimated_forward_bytes(model, tokens, 4) == 4 * estimated_forward_bytes(model, tokens, 1)
    with torch.inference_mode():
        result = residual_patch_sweep(model, tokens, cache, metrics,
                                     index_df=pd.DataFrame(columns=["layer", "pos"]), patch_batch_size=4)
    assert result.values.shape == (2, 0, 2)


def test_single_metric_tensor_return(experiment):
    model, _, tokens, cache, indices, metrics = experiment
    with torch.inference_mode():
        original = upstream.generic_activation_patch(
            model, tokens, cache, metrics["difference"], upstream.layer_pos_patch_setter,
            "resid_pre", index_df=indices, reduce="mean",
        )
        actual = residual_patch_sweep(model, tokens, cache, metrics["difference"], index_df=indices,
                                      reduce="mean", baseline=False, patch_batch_size=4)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, original, atol=1e-5, rtol=1e-5)


def test_source_provenance_matches_installed_original():
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()
    vendored = hashlib.sha256((root / "vendor/transformerlens/patching.py").read_bytes()).hexdigest()
    assert actual == vendored == manifest["source_sha256"]


def test_fp64_noise_batch_only_and_patch_numerics():
    model = load_model(dtype=torch.float64)
    tokens = model.to_tokens(["The capital of France is", "The capital of Japan is"], prepend_bos=False)
    cache = None
    try:
        with torch.inference_mode():
            _, cache = clean_residual_cache(model, tokens)
            embedding = cache["blocks.0.hook_resid_pre"]
            noise = torch.randn(embedding.shape, generator=torch.Generator().manual_seed(17)) * (3 * embedding.std().item())
            noise[:, :3] = 0
            noise[:, 4:] = 0
            def noisy(value, hook):
                return value + noise.repeat(value.shape[0] // 2, 1, 1)
            def metric(logits):
                return logits[:, -1, 6700] - logits[:, -1, 10638]
            with model.hooks(fwd_hooks=[("blocks.0.hook_in", noisy)]):
                baseline = model(tokens)
                expanded = model(tokens.repeat(4, 1))
                assert (expanded - baseline.repeat(4, 1, 1)).abs().max().item() < 1e-8
                serial = upstream.generic_activation_patch(
                    model, tokens, cache, metric, upstream.layer_pos_patch_setter, "resid_pre",
                    index_axis_names=["layer", "pos"], reduce="none", baseline=True)
                batched = residual_patch_sweep(model, tokens, cache, metric, patch_batch_size=4)
                torch.testing.assert_close(batched.values, serial.values, atol=1e-6, rtol=1e-6)
        assert hook_count(model) == 0
    finally:
        if cache is not None:
            cache.cache_dict.clear()
