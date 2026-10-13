# Observed local validation

Environment: Windows CPU, Python 3.12.14, PyTorch 2.14.1+cpu, Transformers 5.19.0, two CPU threads. No CUDA device was available. The model is the public trained Pythia-14m checkpoint pinned in `PROVENANCE.json`; runtime and model files are stored outside the repository.

The upstream serial routine was exercised before the copied core was created. A 6-layer × 5-position × 2-example sweep required 31 forward calls including its baseline. Unpatched direct/empty-hook logits were exactly equal. No forward hooks remained. The original source SHA-256 and installed original file SHA-256 matched.

## Regression checks

39 source regression tests passed in one local run. Five additional output-preservation tests passed in a separate run and were independently confirmed. They cover original/copy/batch agreement, all reduction modes, metric/example axes, grid and explicit index-row order including duplicates, no-intervention equality, prefix controls, cache immutability, repeatability, input and budget rejection, metric and forward failure cleanup, caller-hook preservation, source hashes, and a separate FP64 noisy batch-only control. CLI preflight clamps requested patches to the actual cell count and checks admission before its repeated-input allocation. Existing successful, failed, empty and file output paths are refused before model execution. Full upstream CI was not run; current remote workflow results are separate from these local observations.

## Numerical findings, including the failed case

An initial FP32 noisy demo failed its `atol=1e-5, rtol=1e-5` serial/batch check: one of 120 metric cells differed by `2.765655517578125e-5`. A focused control using the same trained weights found a batch-only logit difference of `1.1110305786132812e-4` and a maximum noisy patched-metric discrepancy of `0.0035996437072753906`. These failures are retained. FP32 noisy batching is not called validated.

FP64 on the same weights reduced the batch-only logit discrepancy to `1.1817213874110166e-11`; patch results agreed exactly. The default demo therefore uses FP64 and requires metric `atol=1e-6, rtol=1e-6` and batch-only logit error at most `1e-8`. It retains the upstream FP32 result storage and discloses that fact.

The successful default report includes two country-replacement conditions and seeded Gaussian embedding noise. For every condition, original/copied and serial/batched result differences were zero, no-intervention logits were exactly equal, identical-corrupted-cache controls had zero effect, and serial/batched rankings matched. The first-position shared-prefix control differed by at most `4.21e-8`, reflecting the disclosed result-storage precision. Repeat runs were exactly equal within each mode. No added hooks remained, and owned caches were cleared.

Metric rankings and corruption-condition rankings differed in these fixed probes. The report includes the raw scores and small-gap diagnostics; the observation does not establish a general rule or a complete circuit. The clean Paris/Tokyo answer probabilities were only about `0.000448` and `0.000362`, so a relative answer-logit preference is not presented as high answer confidence.

## Measured resource comparison

All repetitions are retained in the JSON report; execution order was rotated. For 30 independent cells plus a baseline, serial paths used 31 forward calls and chunk size 4 used 9.

| Condition | Original serial seconds, all runs | Batched seconds, all runs |
| --- | --- | --- |
| Country replacement A | 0.7200, 0.5929, 0.6080 | 0.2117, 0.2117, 0.2100 |
| Country replacement B | 0.5799, 0.5472, 0.5552 | 0.2141, 0.1913, 0.1945 |
| Seeded embedding noise | 0.5340, 0.4947, 0.4783 | 0.1726, 0.1830, 0.1892 |

This fixed short-input CPU experiment was faster with chunking. No universal speed claim is made. The FP64 admission estimate for chunk size 4 was 16,601,600 bytes. Sampled process peak RSS ranged from about 554.6 to 596.5 MiB across runs. The 5 ms samples can miss short peaks; the admission estimate is not a hard process-memory cap.

Wheel and source distributions were built, both retaining the MIT notices. Before the final CLI guards, the installed console command ran outside the checkout with all three FP64 conditions, no leftover hooks and cleared caches. The report passed 18 selector combinations, keyboard interaction, desktop 1440px/mobile 390px, print and no-network checks. Parent visual review confirmed the actual result screenshots. Final frozen-distribution verification is recorded separately in the owner-review evidence; this paragraph does not substitute for it.
