"""Pinned openly downloadable trained model and native-to-upstream cache names."""
import torch
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.bridge_core import build_alias_to_canonical_map

MODEL_ID = "EleutherAI/pythia-14m"
MODEL_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"


def load_model(threads=2, dtype=torch.float32):
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    return TransformerBridge.boot_transformers(
        MODEL_ID, revision=MODEL_REVISION, device="cpu", dtype=dtype,
        trust_remote_code=False,
    ).eval()


def clean_residual_cache(model, tokens):
    names = [f"blocks.{layer}.hook_in" for layer in range(model.cfg.n_layers)]
    logits, cache = model.run_with_cache(tokens, names_filter=names)
    # The upstream generic sweep asks for HT aliases; Bridge stores native keys.
    aliases = build_alias_to_canonical_map(model.hook_dict)
    cache.cache_dict.update({alias: cache.cache_dict[native]
                             for alias, native in aliases.items() if native in cache.cache_dict})
    return logits, cache


def hook_count(model):
    points = {id(value): value for value in model.hook_dict.values()}
    return sum(len(point.fwd_hooks) + len(point.bwd_hooks) for point in points.values())
