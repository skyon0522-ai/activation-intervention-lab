# Third-party material

## TransformerLens

- Source: https://github.com/TransformerLensOrg/TransformerLens
- Commit: `bd55eab6737584685b921fdfcd15f191579e64b1`
- License: MIT, copyright (c) 2022 TransformerLensOrg.
- Unmodified copy: `vendor/transformerlens/patching.py`, SHA-256 in `PROVENANCE.json`.
- Extracted original function: lines 381–566, `vendor/transformerlens/generic_activation_patch.py.txt`.
- Adapted implementation: `src/activation_intervention_lab/_core.py`.
- Reused runtime components: TransformerBridge, native/alias cache mapping, patch setter, metric/result/statistics helpers and hook contexts.
- Change: residual-only shape/index/budget validation and an opt-in independent batch loop. The serial loop, metric validation, output indexing, baseline and reductions remain copied.
- The complete original notice is in `LICENSE` and `vendor/transformerlens/LICENSE`.

The upstream checkout used for baseline execution is retained separately as local evidence. It is not included as a runtime, `.git` directory or model cache in this project.

## Pythia-14m

- Source: https://huggingface.co/EleutherAI/pythia-14m
- Revision: `cf967c0a9a04383db6f7b1108d86b2962634b4ac`
- Model-card license: Apache-2.0.
- Model/config/tokenizer files are downloaded to the user's cache on the first run, not shipped in source or report artifacts.
- The report records file SHA-256 values for the observed pinned checkpoint.
- The project uses trained weights; random initialization is not presented as language ability.

## Research reference

Fred Zhang and Neel Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*, ICLR 2024. https://arxiv.org/abs/2309.16042

The report compares token replacement, seeded Gaussian embedding noise, answer logit difference and answer probability. No paper text, figures, or paper-scale performance results are copied or claimed.
