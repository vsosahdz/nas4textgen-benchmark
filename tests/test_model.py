"""The model builder, and the equivalence that makes a hand-written attention safe.

Grouped-query attention is written by hand because ``GPT2Config`` cannot express it,
and the ``kv`` gene is what produces the objective conflict. Hand-written architecture
is reviewer surface, so it carries two obligations discharged here: with ``kv == h``
the block must be *bit-identical* to unmodified GPT-2, and the KV cache must actually
store the narrow tensors — repeating key/value heads before the cache update would
leave the arithmetic correct while the entire memory objective silently evaporated.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import DynamicCache, GPT2LMHeadModel  # noqa: E402

from nas4textgen.model import build_config, build_model, count_parameters  # noqa: E402
from nas4textgen.space.cost import kv_cache_bytes, parameter_count  # noqa: E402
from nas4textgen.space.genotype import DISTILGPT2, Genotype, enumerate_space  # noqa: E402

pytestmark = pytest.mark.slow


def _transplant(vanilla: GPT2LMHeadModel, gqa: GPT2LMHeadModel, e: int) -> None:
    """Copy an unmodified GPT-2's weights, splitting the fused qkv projection.

    ``Conv1D(nf, nx)`` holds weight of shape ``(nx, nf)``, so the fused ``c_attn``
    splits along its second axis into query, key and value blocks of width ``e``.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in vanilla.state_dict().items():
        if key.endswith("attn.c_attn.weight"):
            base = key[: -len("c_attn.weight")]
            out[base + "q_attn.weight"] = value[:, :e].clone()
            out[base + "kv_attn.weight"] = value[:, e:].clone()
        elif key.endswith("attn.c_attn.bias"):
            base = key[: -len("c_attn.bias")]
            out[base + "q_attn.bias"] = value[:e].clone()
            out[base + "kv_attn.bias"] = value[e:].clone()
        else:
            out[key] = value.clone()
    missing, unexpected = gqa.load_state_dict(out, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all("lm_head" in m for m in missing), f"missing keys: {missing}"


@pytest.mark.parametrize(
    "genotype",
    [
        Genotype(h=4, d_head=32, n=2, ffn=2, kv=4),
        Genotype(h=8, d_head=64, n=2, ffn=2, kv=8),
        Genotype(h=12, d_head=64, n=4, ffn=2, kv=12),
        Genotype(h=16, d_head=128, n=2, ffn=2, kv=16),
    ],
)
def test_kv_equals_h_is_bit_identical_to_unmodified_gpt2(genotype):
    torch.manual_seed(0)
    config = build_config(genotype)
    vanilla = GPT2LMHeadModel(config).eval()
    gqa = build_model(genotype).eval()
    _transplant(vanilla, gqa, genotype.e)

    ids = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        difference = (vanilla(ids).logits - gqa(ids).logits).abs().max().item()
    assert difference == 0.0, f"grouped-query attention diverged from MHA: {difference}"


@pytest.mark.parametrize("kv", [1, 2, 4, 16])
def test_cache_stores_the_narrow_tensors(kv):
    """If this fails, the memory objective is fiction even though the maths is right."""
    genotype = Genotype(h=16, d_head=64, n=2, ffn=2, kv=kv)
    model = build_model(genotype).eval()
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(torch.randint(0, 1000, (1, 32)), past_key_values=cache, use_cache=True)

    assert cache.layers[0].keys.shape[1] == kv

    measured = sum(layer.keys.numel() + layer.values.numel() for layer in cache.layers) * 2
    assert kv_cache_bytes(genotype, context=32) == measured


def test_closed_form_matches_the_anchor():
    assert count_parameters(build_model(DISTILGPT2)) == parameter_count(DISTILGPT2)


def test_closed_form_matches_instantiated_models_across_the_space():
    """A stratified sample spanning the parameter range, per the search-space spec."""
    space = list(enumerate_space())
    order = np.argsort([parameter_count(g) for g in space])
    sample = [space[i] for i in order[:: max(1, len(space) // 55)]][:55]
    assert len(sample) >= 50

    for genotype in sample:
        instantiated = count_parameters(build_model(genotype))
        assert instantiated == parameter_count(genotype), f"mismatch for {genotype}"


def test_generation_runs_under_grouped_query_attention():
    genotype = Genotype(h=12, d_head=64, n=2, ffn=2, kv=2)
    model = build_model(genotype).eval()
    with torch.no_grad():
        out = model.generate(
            torch.randint(0, 1000, (1, 8)),
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=50256,
        )
    assert out.shape[1] == 20


def test_invalid_genotype_is_refused():
    with pytest.raises(ValueError):
        build_model(Genotype(h=8, d_head=64, n=4, ffn=2, kv=6))
