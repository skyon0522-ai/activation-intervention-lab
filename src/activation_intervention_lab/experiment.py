"""Run a fixed trained-model experiment; retain all measurements, not best timings."""
import hashlib
import json
import platform
import threading
import time
from contextlib import nullcontext
from importlib.metadata import version
from itertools import product
from numbers import Integral

import pandas as pd
import psutil
import torch
from huggingface_hub import try_to_load_from_cache
from transformer_lens import patching as upstream

from . import residual_patch_sweep
from .batching import estimated_forward_bytes, validate_residual_sweep
from .model import MODEL_ID, MODEL_REVISION, clean_residual_cache, hook_count, load_model

CLEAN = ["The capital of France is", "The capital of Japan is"]
ANSWERS = [" Paris", " Tokyo"]
CONDITIONS = [
    ("country_swap_A", ["The capital of Italy is", "The capital of China is"], [" Rome", " Beijing"]),
    ("country_swap_B", ["The capital of Germany is", "The capital of Korea is"], [" Berlin", " Seoul"]),
    ("embedding_noise", CLEAN, [" Rome", " Beijing"]),
]


def measure(model, run):
    process = psutil.Process()
    rss_start = process.memory_info().rss
    peak = [rss_start]
    done = threading.Event()
    def sample():
        while not done.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    original_forward = model.forward
    calls = [0]
    def counted_forward(*args, **kwargs):
        calls[0] += 1
        return original_forward(*args, **kwargs)
    model.forward = counted_forward
    sampler.start()
    start = time.perf_counter()
    try:
        result = run()
    finally:
        seconds = time.perf_counter() - start
        done.set()
        sampler.join()
        model.forward = original_forward
    peak[0] = max(peak[0], process.memory_info().rss)
    return result, {"seconds": seconds, "forward_calls": calls[0],
                    "rss_start_bytes": rss_start, "sampled_peak_rss_bytes": peak[0],
                    "sampled_peak_delta_bytes": peak[0] - rss_start,
                    "memory_method": "process RSS sampled every 5 ms; misses shorter allocation peaks"}


def token_ids(model, words):
    encoded = [model.tokenizer.encode(word, add_special_tokens=False) for word in words]
    if any(len(item) != 1 for item in encoded):
        raise ValueError(f"answer alternatives must each be one token: {encoded}")
    return torch.tensor([item[0] for item in encoded])


def metrics_for(answers, alternatives):
    def logit_difference(logits):
        last = logits[:, -1]
        return last.gather(1, answers[:, None]).squeeze(1) - last.gather(1, alternatives[:, None]).squeeze(1)
    def answer_probability(logits):
        return logits[:, -1].softmax(-1).gather(1, answers[:, None]).squeeze(1)
    return {"answer_logit_difference": logit_difference, "answer_probability": answer_probability}


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(*, patch_batch_size=4, repeats=3, threads=2, dtype="float64"):
    if isinstance(patch_batch_size, bool) or not isinstance(patch_batch_size, Integral) or patch_batch_size <= 0:
        raise ValueError("patch_batch_size must be a positive integer")
    if not 1 <= repeats <= 10:
        raise ValueError("benchmark repeats must be between 1 and 10")
    model = load_model(threads, dtype={"float32": torch.float32, "float64": torch.float64}[dtype])
    tolerance = {"atol": 1e-5, "rtol": 1e-5} if dtype == "float32" else {"atol": 1e-6, "rtol": 1e-6}
    clean = model.to_tokens(CLEAN, prepend_bos=False)
    requested_patch_batch_size = patch_batch_size
    patch_batch_size = min(patch_batch_size, model.cfg.n_layers * clean.shape[1])
    patch_indices = pd.DataFrame(product(range(model.cfg.n_layers), range(clean.shape[1])), columns=["layer", "pos"])
    answers = token_ids(model, ANSWERS)
    report = {
        "schema_version": 1, "title": "Residual activation intervention lab",
        "scope": "Fixed next-token probes of trained Pythia-14m; not a language-capability benchmark or complete circuit discovery.",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0",
                  "parameters": sum(p.numel() for p in model.parameters()), "dtype": str(model.cfg.dtype)},
        "upstream": {"commit": "bd55eab6737584685b921fdfcd15f191579e64b1", "license": "MIT"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": version("transformers"),
                        "threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
                        "cpu": platform.processor()},
        "patch_batch_size": patch_batch_size, "requested_patch_batch_size": requested_patch_batch_size, "repeats": repeats,
        "numerical_policy": {"metric_storage_dtype": "torch.float32, inherited from upstream result store",
                             "serial_batched_metric_tolerance": tolerance,
                             "unpatched_batch_only_max_abs_tolerance": 1e-4 if dtype == "float32" else 1e-8,
                             "note": "FP32 noisy probes failed the initial tolerance. FP64 is the default; failure receipts are retained."},
        "admission": {"max_batch_tokens": 4096, "max_batch_bytes": 128 * 1024 * 1024,
                      "estimated_forward_bytes": estimated_forward_bytes(model, clean, patch_batch_size),
                      "expanded_examples": patch_batch_size * clean.shape[0],
                      "note": "Bounds admit tensor sizes. Process RSS is measured separately and is not hard capped."},
        "metric_order": ["answer_logit_difference", "answer_probability"], "conditions": [],
        "paper": {"title": "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods",
                  "url": "https://arxiv.org/abs/2309.16042", "venue": "ICLR 2024",
                  "relationship": "Uses the paper's corruption/metric comparison; does not reproduce its model-scale experiments."},
        "checkpoint_files": {},
    }
    for filename in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        cached = try_to_load_from_cache(MODEL_ID, filename, revision=MODEL_REVISION)
        if isinstance(cached, str):
            report["checkpoint_files"][filename] = {"sha256": digest_file(cached)}
    cache = None
    try:
        with torch.inference_mode():
            clean_logits, cache = clean_residual_cache(model, clean)
            validate_residual_sweep(model, clean, cache, patch_indices, "resid_pre",
                                    upstream.layer_pos_patch_setter, patch_batch_size,
                                    report["admission"]["max_batch_tokens"], report["admission"]["max_batch_bytes"])
            for condition_id, changed, alternative_words in CONDITIONS:
                tokens = model.to_tokens(changed, prepend_bos=False)
                if tokens.shape != clean.shape:
                    raise ValueError("clean/corrupted input shapes must match")
                metrics = metrics_for(answers, token_ids(model, alternative_words))
                noise_info = None
                if condition_id == "embedding_noise":
                    embedding = cache["blocks.0.hook_resid_pre"]
                    std = embedding.std().item()
                    noise = torch.randn(embedding.shape, generator=torch.Generator().manual_seed(17)) * (3 * std)
                    noise[:, :3] = 0
                    noise[:, 4:] = 0
                    def noising_hook(value, hook):
                        if value.shape[0] % noise.shape[0]:
                            raise ValueError("noise requires complete example groups")
                        return value + noise.repeat(value.shape[0] // noise.shape[0], 1, 1)
                    corruption_context = model.hooks(fwd_hooks=[("blocks.0.hook_in", noising_hook)])
                    noise_info = {"seed": 17, "position": 3, "scale": 3, "embedding_std": std,
                                  "tensor_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                                  "location": "first-block residual input, equal to token embeddings for Pythia rotary architecture",
                                  "warning": "Gaussian noising can produce off-distribution activations."}
                else:
                    corruption_context = nullcontext()
                corrupt_cache = None
                try:
                    with corruption_context:
                        caller_hooks = hook_count(model)
                        corrupt_logits, corrupt_cache = clean_residual_cache(model, tokens)
                        no_intervention = model.run_with_hooks(tokens, fwd_hooks=[])
                        baseline = torch.stack([fn(corrupt_logits) for fn in metrics.values()])
                        clean_metric = torch.stack([fn(clean_logits) for fn in metrics.values()])
                        repeated_logits = model(tokens.repeat(patch_batch_size, 1))
                        batch_only_error = (repeated_logits - corrupt_logits.repeat(patch_batch_size, 1, 1)).abs().max().item()
                        del repeated_logits
                        runs = {"original_serial": [], "copied_serial": [], "batched": []}
                        record = {"id": condition_id, "status": "in_progress", "measurements": runs,
                                  "unpatched_batch_only_logits_max_abs": batch_only_error}
                        report["conditions"].append(record)
                        values = {}
                        for repetition in range(repeats):
                            # Rotate order; retain each observation rather than a best timing.
                            modes = list(runs)
                            modes = modes[repetition % 3:] + modes[:repetition % 3]
                            for mode in modes:
                                if mode == "original_serial":
                                    call = lambda: upstream.generic_activation_patch(
                                        model, tokens, cache, metrics, upstream.layer_pos_patch_setter, "resid_pre",
                                        index_axis_names=["layer", "pos"], reduce="none", baseline=True)
                                else:
                                    count = patch_batch_size if mode == "batched" else 1
                                    call = lambda count=count: residual_patch_sweep(
                                        model, tokens, cache, metrics, patch_batch_size=count)
                                result, measurement = measure(model, call)
                                measurement["repetition"] = repetition
                                runs[mode].append(measurement)
                                if mode in values:
                                    torch.testing.assert_close(result.values, values[mode], atol=0, rtol=0)
                                values[mode] = result.values
                        record["serial_vs_batched_max_abs_before_acceptance"] = (values["batched"] - values["original_serial"]).abs().max().item()
                        record["smallest_clean_corrupted_metric_gap"] = (clean_metric - baseline).abs().min(-1).values.tolist()
                        original_rankings = [torch.argsort(metric.mean(-1).flatten(), descending=True).tolist() for metric in values["original_serial"]]
                        batched_rankings = [torch.argsort(metric.mean(-1).flatten(), descending=True).tolist() for metric in values["batched"]]
                        record["serial_vs_batched_rankings_equal"] = original_rankings == batched_rankings
                        torch.testing.assert_close(values["copied_serial"], values["original_serial"], atol=0, rtol=0)
                        torch.testing.assert_close(values["batched"], values["original_serial"], **tolerance)
                        if batch_only_error > report["numerical_policy"]["unpatched_batch_only_max_abs_tolerance"]:
                            raise AssertionError("batch-only numerical control exceeded dtype-specific tolerance")
                        controls = residual_patch_sweep(model, tokens, corrupt_cache, metrics, patch_batch_size=patch_batch_size)
                        same_cache_error = controls.effect().abs().max().item()
                        no_intervention_error = (no_intervention - corrupt_logits).abs().max().item()
                        prefix_error = (values["batched"][:, :, 0, :] - baseline[:, None, :]).abs().max().item()
                        if no_intervention_error != 0 or same_cache_error > 1e-5 or prefix_error > 1e-5:
                            raise AssertionError("intervention controls exceeded their numerical tolerances")
                        if hook_count(model) != caller_hooks:
                            raise AssertionError("sweep left hooks behind")
                        top = corrupt_logits[:, -1].topk(5)
                        effects = values["batched"] - baseline[:, None, None, :]
                        means = effects.mean(-1)
                        rankings = [torch.argsort(metric.flatten(), descending=True).tolist() for metric in means]
                        record.update({
                            "status": "passed",
                            "id": condition_id, "clean_inputs": CLEAN, "corrupted_inputs": changed,
                            "clean_token_ids": clean.tolist(), "corrupted_token_ids": tokens.tolist(),
                            "token_labels": [[model.tokenizer.decode([v]) for v in row] for row in clean.tolist()],
                            "answers": ANSWERS, "alternatives": alternative_words,
                            "answer_token_ids": answers.tolist(), "alternative_token_ids": token_ids(model, alternative_words).tolist(),
                            "noise": noise_info, "clean_metrics": clean_metric.tolist(), "corrupted_metrics": baseline.tolist(),
                            "patched_metrics": values["batched"].tolist(), "patch_effects": effects.tolist(),
                            "rankings_by_mean_effect": rankings, "metric_ranking_differs": rankings[0] != rankings[1],
                            "corrupted_top5_tokens": [[model.tokenizer.decode([i]) for i in row] for row in top.indices.tolist()],
                            "measurements": runs,
                            "checks": {"original_vs_copied_max_abs": (values["original_serial"] - values["copied_serial"]).abs().max().item(),
                                       "serial_vs_batched_max_abs": (values["original_serial"] - values["batched"]).abs().max().item(),
                                       "no_intervention_max_abs": no_intervention_error,
                                       "identical_corrupted_cache_patch_max_abs_effect": same_cache_error,
                                       "shared_prefix_position_zero_max_abs_effect": prefix_error,
                                       "repeatability": "exact equality within each execution mode",
                                       "additional_hooks_after_sweep": hook_count(model) - caller_hooks},
                        })
                finally:
                    if corrupt_cache is not None:
                        corrupt_cache.cache_dict.clear()
            report["corruption_ranking_differs"] = {
                "logit_difference": len({tuple(c["rankings_by_mean_effect"][0]) for c in report["conditions"]}) > 1,
                "probability": len({tuple(c["rankings_by_mean_effect"][1]) for c in report["conditions"]}) > 1,
            }
            report["remaining_hooks"] = hook_count(model)
            if report["remaining_hooks"]:
                raise AssertionError("remaining hooks after experiment")
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["remaining_hooks_after_failure"] = hook_count(model)
        exc.partial_report = report
        raise
    finally:
        if cache is not None:
            cache.cache_dict.clear()
        report["cache_released"] = cache is None or not cache.cache_dict
    return report
