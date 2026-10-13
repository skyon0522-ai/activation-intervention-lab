# Report design and reference

Reader: a developer or researcher who wants to inspect one local causal-intervention experiment and verify the batching change. Intended use: review exact inputs, choose a patch and metric, inspect controls and compare measured resource use. Voice: plain technical English, with uncertainty kept visible.

Inspected references:

- TransformerLens `demos/Main_Demo.ipynb` at `bd55eab6737584685b921fdfcd15f191579e64b1`, the activation-patching section. Adopted order: clean/corrupted task, answer logit difference, layer/position intervention, full patch grid.
- Zhang & Nanda, ICLR 2024, https://arxiv.org/abs/2309.16042. Adopted explicit separation of corruption choice and metric choice, reporting signed effects and avoiding a circuit claim from a localization result.
- TransformerLens original `generic_activation_patch`, lines 381–566, and `layer_pos_patch_setter`. Adopted per-example metric handling, baseline and reductions; the implementation preserves the original routine rather than build a new model.

The report uses native selects, semantic tables, visible keyboard focus and text in every heatmap cell. Color supplements the signed number. The selected patch details update through a polite live region. Model hashes and token IDs sit in a disclosure section. Narrow screens scroll the dense heatmap without reducing text below 16 px. Print and reduced-motion styles are included. There are no remote assets or telemetry.

Editable sources: `report.py` HTML/CSS/JavaScript template and `report.json` experiment data. Artifact revision: SHA-256 of the embedded JSON payload. The report is an experiment artifact, not a published marketing page.

Visual and interaction checks are recorded in `VALIDATION.md` after the rendered output is inspected.
