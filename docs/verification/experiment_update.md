# Contributor and experiment update verification

Verified locally on Windows with CPython 3.12.4. The final core source identity is
`9cf15e340f4d80e1ef2d1a6a589f613882f087a91760e60cc37723bdea867740`.
This is the portable source hash used by `software_identity`; evaluator manifests
also record their own environment identity.

| Check | Result |
|---|---|
| Complete offline suite | 177 passed; two opt-in model tests deselected |
| Ruff over repository, including notebook | Passed |
| mypy | Passed, 47 source files |
| Existing documented CLI commands | Demo, method discovery/checks, static doctor, run, benchmark, calibration and replay passed |
| Contributor fixtures | Both boundary and moving-average templates exercised in copied checkouts; generated implementation/test/card/config discovered and run without registry edits; external plugin card discovery passed |
| Dataset and comparison fixtures | CSV/JSONL validation and mappings, early failure before model loading, CSV/JSONL roundtrips, retained failures/work, paired seed matching, incompatible-run rejection, adaptive uncertainty handling, HTML escaping and stable summary headers passed |
| Build and clean wheel | Wheel/sdist build, Twine checks, separate plugin wheel, clean installation, dataset check, scripted benchmark and custom analysis passed; recorded in `release_checks.json` |
| Source archive contents | Includes example CSV, experiment/dataset/signal/result guides, analysis script and notebook |
| Cached Qwen3 backend/profile regression | Six passed, including real probe isolation, RNG continuation, scores, answer transition and candidate preflight |
| Real REFRAIN and DEER mechanics | REFRAIN completed with answer B and updated its session; induced DEER trial returned valid B and preserved the main prefix |
| New real-model CLI trial | One CSV question, two methods, 16 reasoning/32 answer/16 probe-output token caps; both ran to `answer_incomplete`, were scored incorrect, and appeared in terminal and saved reports |

The model checks used Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`, Torch 2.7.1+cpu and Transformers 4.53.2.
The two-method trial used intentionally small caps to exercise the complete
import/selection/inference/export path and incomplete-result handling. It does not
establish quality or efficiency. The induced DEER prefix was a controlled backend
fixture, not a naturally observed stop. No new model-family or paper-result
reproduction claim follows from these checks.

The HTML's generated content and escaping are covered by offline tests; no browser
automation result is claimed. Public [CI results](https://github.com/YMuskrat/HALT/actions/workflows/ci.yml)
track Linux/Windows and Python 3.11–3.13 for published commits.
