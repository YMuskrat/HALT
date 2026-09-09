"""Single-sequence Transformers execution with isolated full-prefix recompute.

No Torch or Transformers imports occur until ``from_pretrained`` is called.
There is deliberately no shared mutable KV cache. Every model invocation is
charged to the central ledger, including teacher forcing and rejected probes.
"""

from __future__ import annotations

import math
import platform
import re
import time
from typing import Any

from halt.errors import CapabilityError, ConfigurationError
from halt.profiles.qwen3 import (
    FINALIZATION_SUFFIXES,
    QWEN3_MODEL_ID,
    QWEN3_REVISION,
    Qwen3ThinkingProfile,
)
from halt.runtime.budgets import UsageLedger, UsageOperation
from halt.tasks import Task
from halt.types import (
    AnswerSequenceScore,
    BackendInfo,
    CandidateScores,
    EmbedSteps,
    MethodSpec,
    NextTokenFeatures,
    ParsedAnswer,
    Phase,
    PrefixRef,
    ProbeAnswer,
    ProbeRequest,
    ProbeResult,
    RunContext,
    ScoreAnswerSequence,
    ScoreCandidates,
    ScoreFrame,
    ScoreNextToken,
    TokenOutput,
    stable_hash,
)


class TransformersBackend:
    """Reference backend: synchronous, eager decoder-only Qwen3 inference.

    Defaults follow Qwen's pinned generation configuration. Greedy generation
    is opt-in for mechanics and paper recipes; it is not the model card's
    recommended thinking configuration. Model instances are shared read-only;
    prefixes, generators, observations and ledgers are owned by each run.
    """

    def __init__(
        self, model: Any, tokenizer: Any, *, model_id: str, revision: str,
        tokenizer_revision: str, device: str = "cpu", temperature: float = 0.6,
        top_p: float = 0.95, top_k: int = 20, embeddings: Any = None,
    ) -> None:
        import torch
        import transformers

        if not math.isfinite(temperature) or temperature < 0:
            raise ConfigurationError("temperature must be finite and nonnegative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ConfigurationError("top_p must be in (0, 1]")
        if type(top_k) is not int or top_k < 0:
            raise ConfigurationError("top_k must be a nonnegative integer")
        self.torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.profile = Qwen3ThinkingProfile()
        try:
            self.profile.validate(tokenizer, model.config)
        except ValueError as exc:
            raise CapabilityError(str(exc)) from exc
        self.device = device
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.embeddings = embeddings
        eos_ids = model.generation_config.eos_token_id
        self.eos_ids = frozenset(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        capabilities = {
            "visible_reasoning", "token_stream", "step_boundaries", "full_vocab_scores",
            "selected_token_scores", "sequence_scoring", "prefix_branch",
            "exact_prefix_resume", "answer_transition", "cancel_request",
        }
        if embeddings is not None:
            capabilities.add("embeddings")
        hardware: dict[str, Any] = {
            "platform": platform.platform(), "processor": platform.processor(),
            "device": str(model.device), "dtype": str(model.dtype),
            "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
        }
        if str(model.device).startswith("cuda"):
            hardware["gpu"] = torch.cuda.get_device_name(model.device)
        self._info = BackendInfo(
            "transformers", model_id, revision, model_id, tokenizer_revision,
            self.profile.name, frozenset(capabilities), "recompute",
            frozenset({"prefix_branch", "exact_prefix_resume"}),
            {"python": platform.python_version(), "torch": torch.__version__,
             "transformers": transformers.__version__}, hardware,
            execution_settings={
                "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "attention": "eager", "use_cache": False, "profile_version": self.profile.version,
                "signal_scores": "raw_log_probability", "sampling": "per_run_torch_generator",
                "embedding_provider": embeddings.provenance if embeddings is not None else None,
            },
        )

    @classmethod
    def from_pretrained(
        cls, model_id: str = QWEN3_MODEL_ID, *, model_profile: str = "qwen3_thinking",
        revision: str | None = None, tokenizer_revision: str | None = None,
        device: str = "cpu", dtype: str = "float32", cache_dir: str | None = None,
        local_files_only: bool = False, temperature: float = 0.6,
        top_p: float = 0.95, top_k: int = 20, embeddings: Any = None,
    ) -> TransformersBackend:
        if model_profile != "qwen3_thinking":
            raise CapabilityError("Only the audited qwen3_thinking profile is implemented")
        revision = revision or (QWEN3_REVISION if model_id == QWEN3_MODEL_ID else None)
        tokenizer_revision = tokenizer_revision or revision
        if any(value is None or re.fullmatch(r"[0-9a-f]{40}", value) is None
               for value in (revision, tokenizer_revision)):
            raise ConfigurationError("Model and tokenizer revisions must be immutable 40-character commits")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ConfigurationError("dtype must be float32, float16 or bfloat16")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install halt-reasoning[transformers] for the Transformers backend") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=tokenizer_revision, cache_dir=cache_dir,
            local_files_only=local_files_only, trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype), attn_implementation="eager",
        ).to(device)
        assert revision is not None and tokenizer_revision is not None
        return cls(model, tokenizer, model_id=model_id, revision=revision,
                   tokenizer_revision=tokenizer_revision, device=device,
                   temperature=temperature, top_p=top_p, top_k=top_k, embeddings=embeddings)

    @property
    def info(self) -> BackendInfo:
        return self._info

    def validate(self, task: Task, spec: MethodSpec) -> None:
        if spec.finalization_recipe not in FINALIZATION_SUFFIXES:
            raise CapabilityError(f"Unsupported finalization recipe: {spec.finalization_recipe}")
        if "candidate_next_token" in spec.signal_operations:
            self.candidate_ids(task.candidates)

    def candidate_ids(self, candidates: tuple[str, ...]) -> tuple[int, ...]:
        encoded = [self.profile.candidate_tokens(self.tokenizer, candidate) for candidate in candidates]
        if any(len(ids) != 1 for ids in encoded):
            raise CapabilityError("candidate_next_token requires each contextual candidate to be one token; use an explicit sequence-scoring adaptation")
        ids = tuple(ids[0] for ids in encoded)
        if len(set(ids)) != len(ids):
            raise CapabilityError("Candidate labels collide at the exact answer position")
        return ids

    def open(self, context: RunContext, task: Task, ledger: UsageLedger) -> TransformersRun:
        return TransformersRun(self, context, task, ledger)


class TransformersRun:
    def __init__(self, backend: TransformersBackend, context: RunContext,
                 task: Task, ledger: UsageLedger) -> None:
        self.backend, self.context, self.task, self.ledger = backend, context, task, ledger
        self.ids = backend.profile.render(backend.tokenizer, task.render())
        self.prompt_length = len(self.ids)
        self.position = 0
        self.closed = False
        self.transitioned = False
        self.answer_prefix = ""
        self._answer_ids: list[int] = []
        self._answer_logs: list[float] = []
        self.generator = backend.torch.Generator(device=backend.device).manual_seed(context.seed)
        self._text_ids: list[int] = []
        self._decoded = ""
        self._pending_logits, _ = self._forward(self.ids, Phase.PREFILL, "prefill", context.run_id)

    @property
    def prefix(self) -> PrefixRef:
        return PrefixRef(self.context.run_id, "main", self.position,
                         stable_hash(self.ids), self.backend.info.model_revision)

    def _check(self) -> None:
        if self.closed:
            raise RuntimeError("Backend run has already been closed")
        self.ledger.check()

    def _forward(self, ids: list[int], phase: Phase, kind: str, parent: str,
                 *, scored_tokens: int = 0) -> tuple[Any, UsageOperation]:
        self._check()
        model_limit = self.backend.model.config.max_position_embeddings
        if len(ids) > model_limit:
            raise CapabilityError(f"Context exceeds model maximum {model_limit}")
        op = self.ledger.start(
            kind, phase, parent_id=parent, input_tokens=len(ids), scored_tokens=scored_tokens,
            recomputed_prefix_tokens=0 if kind == "prefill" else len(ids),
            forward_calls=1, context_tokens=len(ids),
        )
        started = time.monotonic()
        torch = self.backend.torch
        try:
            with torch.inference_mode():
                inputs = torch.tensor([ids], dtype=torch.long, device=self.backend.device)
                output = self.backend.model(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False,
                    logits_to_keep=1,
                )
                logits = output.logits[0, -1].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Model produced nonfinite raw logits")
            op.status = "completed"
            return logits, op
        except BaseException:
            op.status = "failed"
            raise
        finally:
            op.duration_seconds = time.monotonic() - started

    def _processed(self, logits: Any, temperature: float) -> Any:
        if temperature == 0:
            return logits
        torch = self.backend.torch
        processed = logits / temperature
        if self.backend.top_k:
            threshold = torch.topk(processed, min(self.backend.top_k, processed.numel())).values[-1]
            processed = processed.masked_fill(processed < threshold, -torch.inf)
        if self.backend.top_p < 1:
            values, indices = torch.sort(processed, descending=True)
            remove = torch.softmax(values, dim=-1).cumsum(dim=-1) > self.backend.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            processed = processed.clone()
            processed[indices[remove]] = -torch.inf
        return processed

    def _sample(self, logits: Any, generator: Any, temperature: float) -> tuple[int, float]:
        torch = self.backend.torch
        if temperature == 0:
            token = int(logits.argmax().item())
        else:
            distribution = torch.softmax(self._processed(logits, temperature), dim=-1)
            token = int(torch.multinomial(distribution, 1, generator=generator).item())
        return token, float(torch.log_softmax(logits, dim=-1)[token].item())

    def next_token(self, phase: Phase) -> TokenOutput:
        self._check()
        self.ledger.check(generated_tokens=1, context_tokens=len(self.ids) + 1)
        if self._pending_logits is not None:
            logits, self._pending_logits = self._pending_logits, None
            op = self.ledger.start("sample_prefill", phase)
            op.status = "completed"
        else:
            logits, op = self._forward(
                self.ids, phase, "main_recompute", self.context.run_id,
                scored_tokens=1 if phase == Phase.ANSWERING and self.context.session_reference else 0,
            )
        token, log_prob = self._sample(logits, self.generator, self.backend.temperature)
        self.ledger.generated(op)
        self.ids.append(token)
        self.position += 1
        if phase == Phase.ANSWERING and token not in self.backend.eos_ids:
            self._answer_ids.append(token)
            self._answer_logs.append(log_prob)
        self._text_ids.append(token)
        decoded = self.backend.tokenizer.decode(self._text_ids, skip_special_tokens=False)
        # Incomplete UTF-8 byte tokens are buffered until the next committed token.
        decoded = decoded.rstrip("\ufffd")
        if not decoded.startswith(self._decoded):
            raise ValueError("Tokenizer decoding changed already committed text")
        text, self._decoded = decoded[len(self._decoded):], decoded
        return TokenOutput(text, token, token in self.backend.eos_ids,
                           token == self.backend.profile.end_token_id)

    def _insert(self, ids: list[int], extra: list[int], phase: Phase, parent: str) -> list[int]:
        if not extra:
            return list(ids)
        op = self.ledger.start("insert_context", phase, parent_id=parent,
                               forced_context_tokens=len(extra), context_tokens=len(ids) + len(extra))
        op.status = "completed"
        return list(ids) + extra

    def transition(self, recipe: str) -> None:
        self._check()
        if self.transitioned:
            return
        extra = self.backend.profile.transition(self.backend.tokenizer, self.ids[self.prompt_length:], recipe)
        self.ids = self._insert(self.ids, extra, Phase.ANSWERING, self.context.run_id)
        self._pending_logits = None
        self._text_ids, self._decoded = [], ""
        self.answer_prefix = "\\boxed{" if recipe in {
            "refrain_boxed_v1", "answer_consistency_sentence_common_v1"
        } else ""
        self.transitioned = True

    @property
    def answer_token_log_probs(self) -> tuple[float, ...]:
        """Raw log probabilities overlapping only the parsed answer argument.

        Formatting tokens wholly outside the answer are excluded. A tokenizer
        token straddling an argument boundary is indivisible and is included;
        this token-overlap convention is explicit in the backend audit.
        """
        if not self._answer_ids:
            return ()
        tokenizer = self.backend.tokenizer
        text = self.answer_prefix + tokenizer.decode(self._answer_ids, skip_special_tokens=True)
        matches = list(re.finditer(r"\\boxed\{([^{}]+)\}", text))
        if matches:
            start, end = matches[-1].span(1)
        else:
            normalized = self.task.normalize(text)
            if normalized is None:
                return ()
            start = text.rfind(normalized)
            if start < 0:
                return ()
            end = start + len(normalized)
        previous = len(self.answer_prefix)
        logs = []
        for index, log_prob in enumerate(self._answer_logs):
            current = len(self.answer_prefix) + len(tokenizer.decode(
                self._answer_ids[:index + 1], skip_special_tokens=True))
            if current > start and previous < end:
                logs.append(log_prob)
            previous = current
        return tuple(logs)

    def _branch(self, recipe: str, parent: str) -> list[int]:
        extra = self.backend.profile.transition(self.backend.tokenizer, self.ids[self.prompt_length:], recipe)
        return self._insert(self.ids, extra, Phase.PROBING, parent)

    def _frame(self, prefix: PrefixRef, *, processing: str = "raw",
               coverage: str = "selected_exact") -> ScoreFrame:
        return ScoreFrame(prefix, processing=processing, vocabulary_coverage=coverage,
                          tokenizer_identity=f"{self.backend.info.tokenizer_id}@{self.backend.info.tokenizer_revision}")

    def _sequence(self, ids: list[int], tokens: list[int], parent: str) -> tuple[float, ...]:
        if not tokens:
            raise ValueError("Cannot score an empty answer sequence")
        current = list(ids)
        values: list[float] = []
        for token in tokens:
            logits, _ = self._forward(current, Phase.PROBING, "teacher_force_recompute", parent,
                                      scored_tokens=1)
            values.append(float(self.backend.torch.log_softmax(logits, dim=-1)[token].item()))
            current.append(token)
        return tuple(values)

    def probe(self, request: ProbeRequest, seed: int) -> ProbeResult:
        self._check()
        if request.prefix != self.prefix:
            raise ValueError("Probe references a stale or foreign main prefix")
        first_op = len(self.ledger.operations)
        parent = request.request_id
        torch, tokenizer = self.backend.torch, self.backend.tokenizer
        value: Any
        if isinstance(request, ScoreCandidates):
            ids = self._branch("halt_cot_qwen_common_v1", parent)
            if request.scoring == "candidate_next_token":
                candidate_ids = self.backend.candidate_ids(request.candidates)
                logits, _ = self._forward(ids, Phase.PROBING, "candidate_score", parent,
                                          scored_tokens=len(candidate_ids))
                log_probs = torch.log_softmax(logits, dim=-1)
                scores = tuple(float(log_probs[token].item()) for token in candidate_ids)
            elif request.scoring == "candidate_sequence":
                scores_list = []
                for candidate in request.candidates:
                    tokens = self.backend.profile.candidate_tokens(tokenizer, candidate + request.suffix)
                    logs = self._sequence(ids, tokens, parent)
                    scores_list.append(sum(logs) / len(logs) if request.length_normalize else sum(logs))
                scores = tuple(scores_list)
            else:
                raise ValueError(f"Unsupported candidate scoring {request.scoring!r}")
            branch_prefix = PrefixRef(request.prefix.run_id, parent, request.prefix.position,
                                      stable_hash(ids), request.prefix.model_revision)
            value = CandidateScores(request.candidates, scores, self._frame(branch_prefix),
                                    request.scoring, request.length_normalize)
        elif isinstance(request, ScoreNextToken):
            if request.processing != "raw":
                raise CapabilityError("Scalar signal probes expose raw logits only; sampling processing is recorded separately")
            logits, _ = self._forward(list(self.ids), Phase.PROBING, "next_token_score", parent,
                                      scored_tokens=1)
            logs = torch.log_softmax(logits, dim=-1)
            entropy = None
            if request.include_entropy:
                entropy = float(-(logs.exp() * logs).sum().item())
            value = NextTokenFeatures(
                self._frame(request.prefix, coverage="full"), float(logs.max().item()),
                float(logs[self.backend.profile.end_token_id].item()), entropy,
                {token: float(logs[token].item()) for token in request.token_ids},
            )
        elif isinstance(request, ProbeAnswer):
            value = self._answer_probe(request, seed)
        elif isinstance(request, ScoreAnswerSequence):
            ids = self._branch(request.recipe, parent)
            tokens = list(tokenizer.encode(request.answer + request.suffix, add_special_tokens=False))
            value = AnswerSequenceScore(self._sequence(ids, tokens, parent), request.answer,
                                         request.suffix, request.length_normalize)
        elif isinstance(request, EmbedSteps):
            if self.backend.embeddings is None:
                raise CapabilityError("No embedding provider configured")
            provider = self.backend.embeddings
            op = self.ledger.start("embed_steps", Phase.PROBING, parent_id=parent,
                                   embedding_calls=1, input_tokens=provider.input_tokens(request.steps))
            started = time.monotonic()
            try:
                value = provider.embed(request.steps)
                op.status = "completed"
            except BaseException:
                op.status = "failed"
                raise
            finally:
                op.duration_seconds = time.monotonic() - started
        else:
            raise CapabilityError(f"Unsupported probe type {type(request).__name__}")
        operations = tuple(op.operation_id for op in self.ledger.operations[first_op:])
        return ProbeResult(parent, request.prefix, value, operation_ids=operations)

    def _answer_probe(self, request: ProbeAnswer, seed: int) -> ParsedAnswer:
        tokenizer, torch = self.backend.tokenizer, self.backend.torch
        deer = request.recipe == "deer_qwen3_greedy_v1"
        if deer:
            # Match the pinned greedy Qwen3 branch: remove the detected Wait
            # from the trial prefix, preserve it unchanged in the main branch.
            ids = list(self.ids)
            text = tokenizer.decode(ids, skip_special_tokens=False)
            if not text.endswith("Wait"):
                raise ValueError("DEER trial requires an exact terminal Wait trigger")
            trigger = tokenizer.encode("Wait", add_special_tokens=False)
            if ids[-len(trigger):] != trigger:
                raise CapabilityError("DEER Wait trigger is not an exact removable token suffix")
            ids = ids[:-len(trigger)]
            extra = tokenizer.encode("\n**Final Answer**\n\\boxed", add_special_tokens=False)
            ids = self._insert(ids, list(extra), Phase.PROBING, request.request_id)
        else:
            ids = self._branch(request.recipe, request.request_id)
            if request.induction:
                ids = self._insert(ids, list(tokenizer.encode(request.induction, add_special_tokens=False)),
                                   Phase.PROBING, request.request_id)
        generator = torch.Generator(device=self.backend.device).manual_seed(seed)
        generated: list[int] = []
        log_probs: list[float] = []
        complete, end_found = False, not deer
        for _ in range(request.max_work.generated_tokens):
            self.ledger.check(generated_tokens=1, context_tokens=len(ids) + 1)
            logits, op = self._forward(ids, Phase.PROBING, "answer_probe_recompute", request.request_id,
                                       scored_tokens=1 if request.score_answer or deer else 0)
            token, log_prob = self._sample(logits, generator, request.temperature)
            self.ledger.generated(op)
            ids.append(token)
            generated.append(token)
            log_probs.append(log_prob)
            if deer and token == self.backend.profile.end_token_id:
                end_found = complete = True
                break
            if token in self.backend.eos_ids:
                complete = True
                break
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Boxed recipes end their induction just before the argument.
        if ("boxed" in request.recipe or "answer_convergence" in request.recipe
                or "answer_consistency" in request.recipe or deer):
            if raw.startswith("{"):
                raw = "\\boxed" + raw
            elif request.recipe in {"refrain_boxed_v1", "answer_consistency_sentence_common_v1"}:
                raw = "\\boxed{" + raw
        answer = self.task.normalize(raw)
        return ParsedAnswer(raw, answer, answer is not None and complete, complete,
                            tuple(log_probs), end_found)

    def cancel(self) -> None:
        self.ledger.cancellation.cancel()

    def close(self) -> None:
        self._pending_logits = None
        self.ids.clear()
        self._text_ids.clear()
        self.closed = True
