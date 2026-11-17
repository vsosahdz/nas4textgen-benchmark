"""Closed-form cost objectives.

Both objectives are exact functions of the genotype, computable without
instantiating a model. That is what makes the cost front enumerable and the
verification tier reproducible on a machine with no GPU.

The parameter count is validated two ways: against the real DistilGPT2 checkpoint's
count of 81,912,576, and against instantiated models across a stratified sample of
the space. A closed form that disagrees with the model it claims to describe is the
defect that let the prior work report parameter counts for architectures that could
not exist.

**Why two objectives and not three.** FLOPs per token was designed in and removed:
measured over the enumerated space, its rank correlation with parameter count is
+0.965. That is the ``FLOPs ~ 2 * params`` identity — the same objective under a
different name. Decode latency is measured rather than modelled, because it is
memory-bound and any closed form would be fiction.

**Where the conflict lives.** Cost objectives do not conflict among themselves:
parameters, memory and FLOPs all shrink as a model shrinks, so the cost-only Pareto
front is a single point. The conflict appears at *matched parameter count*, where
KV-cache memory varies by up to 192x — see design.md D4.
"""

from __future__ import annotations

from .genotype import MAX_POSITION, REFERENCE_CONTEXT, VOCAB_SIZE, Genotype

#: Bytes per stored element in the KV cache. Half precision is what inference uses.
KV_DTYPE_BYTES = 2


def block_parameters(g: Genotype) -> int:
    """Parameters in one transformer block.

    GPT-2 block layout with grouped-query attention substituted for multi-head
    attention. Everything except the key and value projections is unchanged from the
    anchor architecture; ``kv == h`` recovers standard multi-head attention exactly.

    Layer norms carry both a weight and a bias, so ``2 * e`` each. The feed-forward
    network is the GPT-2 two-matrix form with biases, not a three-matrix gated form.
    """
    e = g.e
    kv_width = g.kv_width
    inter = g.intermediate

    # Attention: query and output projections at full width; key and value
    # projections narrowed to the KV head count. This is the only departure from the
    # anchor block, and it is what makes memory depend on the head configuration.
    q_proj = e * e + e
    k_proj = e * kv_width + kv_width
    v_proj = e * kv_width + kv_width
    o_proj = e * e + e

    # Feed-forward: expand to `inter`, project back.
    fc = e * inter + inter
    proj = inter * e + e

    # Two layer norms per block, weight and bias each.
    norms = 2 * (2 * e)

    return q_proj + k_proj + v_proj + o_proj + fc + proj + norms


def parameter_count(g: Genotype) -> int:
    """Total trainable parameters.

    The language-modelling head shares weights with the token embedding, as in the
    anchor architecture, so it contributes nothing.
    """
    token_embedding = VOCAB_SIZE * g.e
    position_embedding = MAX_POSITION * g.e
    final_norm = 2 * g.e
    return token_embedding + position_embedding + g.n * block_parameters(g) + final_norm


def embedding_share(g: Genotype) -> float:
    """Fraction of the parameter count held by the token embedding table.

    Reported because it is the quantity that made the prior objective degenerate: at
    ``e = 8`` the token embedding was 98.6% of the count, so "minimise parameters"
    reduced to "minimise ``e``", a single gene out of seven.
    """
    return (VOCAB_SIZE * g.e) / parameter_count(g)


def kv_cache_bytes(
    g: Genotype,
    context: int = REFERENCE_CONTEXT,
    dtype_bytes: int = KV_DTYPE_BYTES,
) -> int:
    """Bytes held by the key/value cache at a given context length.

    Two tensors — keys and values — per layer, each of shape
    ``(context, kv * d_head)``. This is the memory that actually constrains
    deployment on a small device, and unlike the parameter count it is not monotone
    in model size, which is why the two objectives conflict.

    ``context`` is explicit and never defaulted silently in reports: a KV-cache
    figure without its context length cannot be interpreted.
    """
    return 2 * g.n * context * g.kv_width * dtype_bytes


def objectives(g: Genotype, context: int = REFERENCE_CONTEXT) -> dict[str, int]:
    """The cost objectives as a mapping, for tabulation. Both are minimised."""
    return {
        "params": parameter_count(g),
        "kv_bytes": kv_cache_bytes(g, context=context),
    }


def forward_flops_per_token(g: Genotype) -> int:
    """Non-embedding forward FLOPs for one token. What wall clock actually tracks.

    The embedding lookup is a gather, not arithmetic, so it does not appear here — which is
    the whole reason this ranks the space differently from ``parameter_count``. In this
    space the parameter count is dominated by the 50,257-row embedding matrix, so ordering
    architectures by it orders them by width; ordering by this orders them by cost.
    """
    return g.n * (4 * g.e * g.e + 2 * g.e * g.intermediate)


def activation_width(g: Genotype) -> int:
    """Depth times width: what peak training memory scales with, at fixed batch and block."""
    return g.n * g.e
