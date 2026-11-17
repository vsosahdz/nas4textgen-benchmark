"""Build a model from a genotype.

The block family is GPT-2 — the anchor architecture — with one substitution:
multi-head attention becomes grouped-query attention, so the number of key/value
heads is searchable. Everything else (learned positional embeddings, LayerNorm, the
two-matrix GELU feed-forward, weight-tied head) is unchanged from the anchor.

That choice was measured rather than assumed. Against a LLaMA-style block, GPT-2 with
grouped-query attention gives a marginally better objective conflict
(rank correlation between parameters and KV-cache memory of +0.661 against +0.672)
at the same space size. Both figures were taken on the space as it stood before
multi-query attention was admitted; the current space correlates at +0.615, which the
structural gate re-derives on every run, and the comparison above is kept at the values
the decision was actually made on. The GPT-2 block also reproduces DistilGPT2 at its exact
parameter count of 81,912,576. In a LLaMA-style space the anchor does not exist: the same shape costs
95.2M because a gated feed-forward has three matrices instead of two, so the claim
"anchored on DistilGPT2" would be fiction.

**Where the memory saving actually comes from.** The cache must store the narrow
key/value tensors. Key/value heads are therefore repeated to the query head count
*after* the cache update, never before — repeating first would populate the cache
with ``h`` heads and the entire memory objective would silently evaporate while the
arithmetic stayed correct.

``kv == h`` must reduce to ordinary multi-head attention exactly. That equivalence is
asserted against an unmodified GPT-2 in the test suite, which is what makes a
hand-written attention acceptable here.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.pytorch_utils import Conv1D

from .space.genotype import MAX_POSITION, VOCAB_SIZE, Genotype


def repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand ``(batch, kv_heads, seq, head_dim)`` to the query head count.

    Each key/value head is shared by ``groups`` query heads. Applied after the cache
    update so that only the narrow tensors are stored.
    """
    if groups == 1:
        return states
    batch, kv_heads, seq, head_dim = states.shape
    states = states[:, :, None, :, :].expand(batch, kv_heads, groups, seq, head_dim)
    return states.reshape(batch, kv_heads * groups, seq, head_dim)


class GQAAttention(GPT2Attention):
    """GPT-2 self-attention with a separate, narrower key/value projection."""

    def __init__(self, config: GPT2Config, layer_idx: int | None = None) -> None:
        super().__init__(config, is_cross_attention=False, layer_idx=layer_idx)

        self.num_kv_heads = int(config.num_key_value_heads)
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_kv_heads})"
            )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.kv_width = self.num_kv_heads * self.head_dim

        # Replace the fused qkv projection with a full-width query projection and a
        # narrowed key/value projection. Conv1D(nf, nx) holds weight of shape
        # (nx, nf), matching the anchor's layout so weights can be transplanted.
        del self.c_attn
        self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        self.kv_attn = Conv1D(2 * self.kv_width, self.embed_dim)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward

        query_states = self.q_attn(hidden_states)
        key_states, value_states = self.kv_attn(hidden_states).split(self.kv_width, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        # The cache stores the NARROW tensors. This ordering is the whole point.
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights


def build_config(g: Genotype) -> GPT2Config:
    """Config for a genotype. Dropout is zero: it is not part of the search space.

    The prior work searched two dropout rates, which are inactive at inference and
    move neither objective, and which inflated the declared space by a factor of 400.
    """
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_POSITION,
        n_embd=g.e,
        n_layer=g.n,
        n_head=g.h,
        n_inner=g.intermediate,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        activation_function="gelu_new",
    )
    config.num_key_value_heads = g.kv
    config.tie_word_embeddings = True
    return config


def build_model(g: Genotype) -> GPT2LMHeadModel:
    """Instantiate the model for a genotype, with grouped-query attention installed."""
    if not g.is_valid():
        raise ValueError(f"invalid genotype: {g}")
    config = build_config(g)
    model = GPT2LMHeadModel(config)
    for layer_idx, block in enumerate(model.transformer.h):
        block.attn = GQAAttention(config, layer_idx=layer_idx)
    model.tie_weights()
    return model


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters, counting a tied weight once.

    ``lm_head`` shares storage with the token embedding, so naive summation over
    ``parameters()`` would double-count it.
    """
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total
