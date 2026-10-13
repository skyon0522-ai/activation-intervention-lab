# Operating and numerical details

Run a small trained language model locally, compare clean and corrupted inputs, and inspect the effect of replacing one residual-stream activation. The CLI writes an interactive, self-contained HTML report and machine-readable JSON. It works on CPU and needs no API keys or paid service.

The core is copied from [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens/tree/bd55eab6737584685b921fdfcd15f191579e64b1), under its MIT license. The opt-in change groups independent residual patches into bounded example batches. It preserves the original patch indices, example and metric axes, metric reductions, baseline, and `PatchingResult` format. `patch_batch_size=1` retains the copied serial loop.

The measurements in VALIDATION.md describe local execution. Consult the repository's Actions tab for current CI results.

## Install and run

Python 3.12 is the observed environment. Git is required to install the exact TransformerLens commit. Use a short virtual-environment path on Windows: deep paths can fail while installing PyTorch.

```sh
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]" -c constraints-verified.txt
```

Run from any directory, including outside this checkout:

```sh
activation-lab --output ./activation-report --patch-batch-size 4 --benchmark-repeats 3
```

The first run downloads the public trained [`EleutherAI/pythia-14m`](https://huggingface.co/EleutherAI/pythia-14m) checkpoint, pinned to `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. Its Apache-2.0 weights stay in the user cache, outside the repository and output report. No credentials are used. Later runs can require that cached checkpoint:

```sh
activation-lab --offline --output ./activation-report-offline
```

Open `activation-report/report.html` in a browser. The report contains all data and JavaScript; it makes no network requests. The companion `report.json` records exact inputs, token IDs, model revision and file hashes, patch scores, controls, every benchmark repetition, and environment details. `--cache-dir` selects a separate model cache; `--threads` controls CPU threads.

The default demo loads the learned weights in **FP64**. A noisy FP32 probe produced a serial/batched discrepancy, so the project does not weaken its initial FP32 tolerance to call that run a success. `--dtype float32` enables a diagnostic experiment; a failed numerical check writes `failure.json` and exits unsuccessfully. The upstream result tensor stores metrics in FP32 even for an FP64 model; this inherited behavior is recorded explicitly. The API's default patch count remains one.

## What the experiment measures

The fixed example probes country/capital next-token scores for France/Paris and Japan/Tokyo. Two symmetric token-replacement conditions change the country. A third condition adds seeded Gaussian noise to the country token's first-block residual input, which is the token embedding for this rotary Pythia model. The noisy condition has the same text and an explicit intervention record.

The two metrics are answer logit minus alternative logit, and the softmax probability of the single answer token. A probability is not a logit difference. The report shows clean, corrupted and patched scores separately, allows metric/example selection, and shows the signed change from the corrupted score for each cell.

Pythia-14m has trained weights and 14,364,928 parameters. Its small size limits its language capability. The probes do not establish factual-answer accuracy, a complete causal circuit, or results on larger models. A patch effect also depends on the chosen input, corruption and metric.

The methodological reference is [Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*, ICLR 2024](https://arxiv.org/abs/2309.16042). The demo adopts the corruption/metric comparison and negative-effect reporting. It does not reproduce the paper's experiments. Gaussian noise can produce off-distribution activations.

## API

```python
from activation_intervention_lab import residual_patch_sweep

result = residual_patch_sweep(
    model, corrupted_tokens, clean_cache,
    {"logit_difference": metric_per_example},
    patch_batch_size=4,  # default API path is serial: 1
    reduce="none", baseline=True,
)
# result.values: [metric, layer, position, example]
```

Each metric receives one original `[example, position, vocabulary]` logits slice, even when several patches share a forward pass. A closure with per-example answer IDs therefore keeps its original example alignment. An explicit `index_df` must contain integer `layer` and `pos` columns in that order; the result uses flat index-row order, including duplicates. `reduce="mean"` averages examples after evaluation. `reduce="scalar"` requires a scalar metric for each independent patch.

`activation_intervention_lab.model.clean_residual_cache` maps current Bridge-native block-input cache keys to the aliases required by the upstream generic core. It reuses the same tensors and does not silently broadcast, reorder or merge examples. The caller owns an API-supplied cache. The CLI clears its own caches in `finally` blocks. Hook contexts remove only the experiment's added hooks after success or failure, and preserve caller hooks.

## Bounds and verification

The default admission limits are 4,096 expanded input tokens and a 128 MiB forward tensor-size estimate. The estimate includes logits, residual work and attention scores. These bounds reject oversized configurations before hooks are attached. They are **not a hard cap on process RSS**. The report separately measures process RSS every 5 ms and discloses missed short peaks and allocator effects.

Serial and batched forwards can use different numerical kernels. Absolute/relative agreement, batch-only numerical controls, rankings and raw metrics must be inspected together; small effects and small clean/corrupted gaps need particular care. All observed timings remain in the report. Fewer forward calls do not automatically mean faster execution.

```sh
python -m pytest tests -q
python -m build
```

Tests use the pinned trained checkpoint, so the first test run also needs its public download. They check original/copy agreement, independent batching and duplicate index rows, metric/example axes and reductions, no-intervention equality, shared-prefix controls, cache immutability, invalid shapes/positions/budgets, repeatability, and hook cleanup after injected failures. The local evidence and final observed results are summarized in [VALIDATION.md](../VALIDATION.md).

## Provenance and licenses

[PROVENANCE.json](../PROVENANCE.json) records the exact upstream commit, full-source SHA-256 and extracted-function range/hash. `vendor/transformerlens/patching.py` is an unchanged source copy with the original [MIT notice](../vendor/transformerlens/LICENSE). `_core.py` is the adapted source; [THIRD_PARTY.md](../THIRD_PARTY.md) explains the copied code and the separately downloaded model. Runtime environments, model weights and caches are excluded from publication contents.
