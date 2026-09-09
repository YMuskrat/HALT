# Transformers reference backend

The implementation owns single-sequence decoding, prefix probes and the answer
phase. It supports the decoder-only `qwen3_thinking` profile. The default model
and tokenizer are both pinned to `Qwen/Qwen3-0.6B` commit
`c1899de289a04d12100db370d81485cdf75e47ca`. Selecting another model requires explicit
immutable model/tokenizer revisions and profile validation; it does not imply
that model has passed the recorded smoke tests.

Primary sources audited:

- [Pinned Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md)
- [Pinned tokenizer and native chat template](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)
- [Pinned generation defaults](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/generation_config.json)
- [Transformers 4.53.2 generation documentation](https://huggingface.co/docs/transformers/v4.53.2/en/main_classes/text_generation)
- [Transformers 4.53.2 Qwen3 forward implementation](https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/qwen3/modeling_qwen3.py)

## Execution and score semantics

Importing `halt` or `halt.backends.TransformersBackend` does not import Torch or
load a model. Explicit `from_pretrained` loads a model with remote code disabled,
eager attention, the requested dtype, and `eval()`. Calls use inference mode,
an all-one attention mask, implicit sequential positions, `use_cache=False`,
and `logits_to_keep=1` to avoid storing every position's vocabulary logits.

The native chat template is invoked with `enable_thinking=True`. At this
revision the template leaves the model to generate the opening thinking tag.
The profile validates that `</think>` is exactly token 151668. Model EOS IDs
remain distinct. A forced closure appends the delimiter once followed by the
named recipe's suffix. Injected tokens are input context, never generated work.

Main sampling defaults to temperature 0.6, top-p 0.95, top-k 20, using a per-run
Torch generator. Temperature zero explicitly selects greedy decoding. The
model card advises sampling for thinking; small greedy tests are mechanics
checks, not recommended benchmark settings.

Signals always use raw model logits before temperature or truncation. Full
entropy uses the whole vocabulary and is measured in nats. The selected-token
margin uses raw top log probability minus raw reasoning-end log probability,
at the exact requested prefix. The top includes every vocabulary token,
including the delimiter. Requests for processed signal scores fail explicitly;
the processed sampling distribution is never passed off as raw full-vocabulary
confidence. Full score tensors stay on device, and only requested scalar
features or candidate values leave the model device.

Candidate scoring closes a separate prefix and appends
`\nTherefore, the final answer is`. Each candidate continuation includes its
leading space. Contextual tokenization must preserve the answer cue; next-token
scoring requires one distinct token per candidate. For this pinned tokenizer,
`A`, `B`, `C`, `D` map to 362, 425, 356, 422 at this answer position. The explicit
sequence variant teacher-forces the complete candidate and termination suffix;
optional length normalization remains distinct from candidate probability.

## Branches and accounting

Every probe re-prefills an independent copy of exact token IDs. It never mutates
main token IDs, a main RNG or a shared KV cache. Main generation pauses during a
probe, and its next sample has the same RNG state whether a discarded probe was
executed or omitted. No parallel decoding or cache speedup is claimed.

Every forward call records its complete input length, repeated prefix work,
scored positions, duration and outcome. Every sampled output, including EOS,
is charged to the reasoning, probe or answer phase. Teacher forcing generates
zero output tokens. The first prefill is charged separately from its first
sampling operation. Input insertion is recorded separately and charged as model
input when it reaches a forward call. Probe operations point to their request
ID. Scoring cost and repeated prefix cost are visible even for rejected trials.

Cancellation and deadlines are checked between synchronous forward calls.
An in-flight CPU/GPU forward cannot be preempted. Closing a run releases prefix,
pending logits and decode buffers; the explicitly constructed shared model
stays loaded for subsequent requests.

## Named finalization recipes

| Recipe | Prefix behavior |
| --- | --- |
| `halt_default_v1` | Close thinking once, append two newlines and `Final answer:` |
| `halt_cot_qwen_common_v1` | Close thinking, append explicit answer-candidate cue |
| `answer_consistency_sentence_common_v1` | Close thinking, append a boxed-answer opening |
| `refrain_boxed_v1` | Close thinking, append `Final Answer: \\boxed{` |
| `deer_qwen3_greedy_v1` | Trial only: remove exact terminal `Wait` from branch, leave thinking open, append `**Final Answer**` and boxed cue |

DEER's trial records all generated raw token log probabilities, including the
closing delimiter, and whether that delimiter actually appeared. It uses its
declared greedy trial settings; rejected trials leave the original `Wait` on
the main trajectory. A tokenizer token containing `Wait` plus additional text
is not silently removed: this backend rejects that unsupported branch shape.

Adaptive-session reward scoring uses raw log probabilities for tokens
overlapping the parsed final-answer argument; formatting-only tokens and EOS
are excluded. A token crossing the argument boundary is indivisible and is
included by the documented overlap convention. Normalization that changes the
literal answer text can leave no identifiable token span; it yields an empty
score rather than an invented reward. Scoring positions are charged on final
answer decoding when an adaptive session requests this consumer.

## Optional embeddings

`halt.signals.embeddings.SentenceTransformerEmbeddings` explicitly loads
`sentence-transformers/all-MiniLM-L6-v2` at
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It emits L2-normalized vectors and
records the encoder's native maximum sequence length/truncation. Similarity is
cosine via inner product. Embedding input tokens and calls are recorded. Without
an explicit provider the backend does not advertise `embeddings`, and REFRAIN's
embedding requirement fails preflight. See the method audit for the scope of
REFRAIN's paper relationship; merely selecting this encoder establishes no
algorithm fidelity.

## Running the checks

```powershell
python -m pip install -e '.[transformers,dev]'
python examples/real_model.py --cache-dir .model-cache
$env:HALT_RUN_MODEL_TESTS = '1'
$env:HALT_MODEL_CACHE = '.model-cache'
python -m pytest tests/test_transformers.py -q
```

After the first download, add `--local-files-only` to the example. The model
tests always require cached files and cannot silently download. No network or
GPU is required for the other profile tests. See
[recorded model evidence](backend_verification.md) for actual runs and limits.
