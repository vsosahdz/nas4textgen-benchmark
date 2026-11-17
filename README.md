# An Exhaustively Measured Benchmark for Multi-Objective NAS in Text Generation

The measurements, the exact Pareto fronts and the analysis code behind the article of the
same name, archived together so that any number in the paper can be recomputed from the
data it came from.

All 1,099 architectures of a DistilGPT2-derived design space were trained once, to an
identical budget of 100 million tokens, under one protocol. The Pareto front over held-out
perplexity, parameter count and key-value cache size is therefore computed from
measurements rather than estimated from a sample, which is what lets a search method, a
regressor or a sampling design be scored as a distance to a known answer.

## What is here

| path | holds |
|---|---|
| `artifacts/benchmark/` | the benchmark table and its summary |
| `artifacts/manifests/` | one manifest per training run, 3,563 of them |
| `artifacts/campaign/` | what the campaign consumed, from allocated node time |
| `artifacts/characterisation/`, `artifacts/search/`, `artifacts/surrogate/` | the analyses the article reports |
| `src/nas4textgen/` | the code that produced and reads all of it |
| `tests/` | the suite, including the checks that hold the artifacts to the article |
| `docs/REPLICATION.md` | what reproduces at which cost, stage by stage |

## Reproducing the analysis

The verification tier needs any CPU, no GPU and no cloud account, and runs in under ten
minutes. It reassembles the benchmark table from the published run manifests and recomputes
the enumeration, the cost front, the sampling designs, the regressor study, the
characterisation and every table in the article, comparing each against what was published.

```bash
pip install -r requirements/base.txt
python -m nas4textgen.cli verify
```

`docs/REPLICATION.md` states the three heavier tiers, what each one needs and what each one
establishes. The heaviest retrains the whole census and needs a GPU cluster.

## Citing this

Cite the article for the findings and this archive for the data. `CITATION.cff` carries the
authorship; the Zenodo deposit carries the DOI for the exact revision the article reports.

## Licence

Apache-2.0. See `LICENSE`.
