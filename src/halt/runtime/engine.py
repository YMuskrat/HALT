from __future__ import annotations

import copy
import dataclasses
import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from halt.backends.base import Backend, BackendRun
from halt.errors import (
    BudgetExceeded,
    Cancelled,
    CapabilityError,
    ConfigurationError,
    MethodError,
    ReplayUnavailable,
)
from halt.methods.base import HaltMethod
from halt.provenance import software_identity
from halt.runtime.boundaries import Segmenter
from halt.runtime.budgets import CancellationToken, UsageLedger
from halt.tasks import Task
from halt.types import (
    Abstain,
    Budget,
    CandidateScores,
    Continue,
    EventKind,
    Finalize,
    Observation,
    ParsedAnswer,
    Phase,
    PrefixRef,
    ProbeFailure,
    ProbeResult,
    RequestSignals,
    ReturnAnswer,
    RunContext,
    RunResult,
    RunStatus,
    StopInfo,
    stable_hash,
    to_data,
)


def method_configuration(method: HaltMethod) -> dict[str, Any]:
    if hasattr(method, "configuration"):
        return dict(method.configuration())
    if dataclasses.is_dataclass(method):
        return dataclasses.asdict(method)
    return {k: v for k, v in vars(method).items() if not k.startswith("_")}


def probe_random_seed(base_seed: int, request: Any, invocation: int) -> int:
    """Run IDs label work; they must not change seeded sampling across equivalent runs."""
    settings = to_data(request)
    settings.pop("request_id")
    settings["prefix"].pop("run_id")
    return int(stable_hash([base_seed, type(request).__name__, settings, invocation])[:15], 16)


class Runtime:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    @staticmethod
    def preflight(backend: Backend, task: Task, method: HaltMethod, budget: Budget) -> dict[str, Any]:
        spec = method.spec
        if spec.api_version != "1":
            raise CapabilityError(f"plugin API {spec.api_version} unsupported; expected 1")
        requirements = spec.requirements
        info = backend.info
        missing = requirements.capabilities - info.capabilities
        problems = sorted(missing)
        for alternative in requirements.alternatives:
            if not alternative & info.capabilities:
                problems.append("one of: " + ", ".join(sorted(alternative)))
        if not requirements.allow_emulated and requirements.capabilities & info.emulated_capabilities:
            problems.append("native execution required; selected backend emulates requested capabilities")
        if requirements.profiles and info.profile not in requirements.profiles:
            problems.append(f"profile must be one of {requirements.profiles}; got {info.profile}")
        if problems:
            raise CapabilityError(f"{spec.method_id} incompatible with {info.name}/{info.model_id}: " + "; ".join(problems))
        if budget.max_answer_tokens == 0 and not (spec.can_return_answer or spec.can_abstain):
            raise ConfigurationError("zero answer allowance requires a return-answer or abstention recipe")
        if spec.probe_failure_policy not in {"continue", "fail"}:
            raise ConfigurationError("probe_failure_policy must be continue or fail")
        Segmenter(spec.boundary)
        validator = getattr(method, "validate_task", None)
        if validator is not None:
            validator(task)
        backend.validate(task, spec)
        return {"compatible": True, "method": to_data(spec), "backend": to_data(info),
                "budget": to_data(budget), "configuration": method_configuration(method)}

    def stream(self, task: Task, method: HaltMethod, budget: Budget, *, seed: int = 0,
               capture: str = "none", trace_path: str | Path | None = None,
               session: Any = None, run_id: str | None = None,
               cancellation: CancellationToken | None = None) -> Iterator[Observation]:
        if capture not in {"none", "metadata", "full"}:
            raise ConfigurationError("capture must be none, metadata, or full")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ConfigurationError("seed must be an integer in [0, 2**63)")
        if trace_path is not None and capture == "none":
            raise ConfigurationError("trace_path requires explicit metadata or full capture")
        self.preflight(self.backend, task, method, budget)
        run_id = run_id or uuid.uuid4().hex
        config = method_configuration(method)
        configured_spec = method.spec
        software = software_identity()
        active = copy.deepcopy(method)
        ledger = UsageLedger(run_id, budget, cancellation)
        started = time.monotonic()
        context = RunContext(run_id, configured_spec.method_id, stable_hash(config), task.visible(),
                             self.backend.info, seed, seed ^ 0x5DEECE66D, budget,
                             getattr(session, "session_id", None))
        handle: BackendRun | None = None
        trace_file = None
        event_index = 0
        phase = Phase.PREFILL
        last_event_id: str | None = None
        stop: StopInfo | None = None
        status = RunStatus.BACKEND_ERROR
        error: str | None = None
        answer: str | None = None
        answer_text = ""
        selected_log_probs: tuple[float, ...] = ()
        probes: dict[str, ProbeResult] = {}
        probe_seeds: list[dict[str, Any]] = []
        per_prefix: dict[str, int] = {}
        durations: dict[str, float] = {"prefill": 0.0, "reasoning": 0.0, "probes": 0.0,
                                      "finalization": 0.0, "controller": 0.0}
        session_initial = session.to_dict() if session is not None else None
        session_diagnostics: Any = None
        propagating = False
        fallback_prefix = PrefixRef(run_id, "main", 0, stable_hash(task.render()), self.backend.info.model_revision)

        def emit(kind: EventKind, payload: Any = None, parent: str | None = None) -> Observation:
            nonlocal event_index, last_event_id
            event = Observation(f"{run_id}:event:{event_index}", kind, time.monotonic()-started,
                                phase, handle.prefix if handle and phase != Phase.FINISHED else fallback_prefix, payload, parent)
            event_index += 1
            last_event_id = event.event_id
            if trace_file is not None:
                record = to_data(event)
                record["payload_type"] = type(payload).__name__
                if isinstance(payload, ProbeResult):
                    record["value_type"] = type(payload.value).__name__
                if capture == "metadata" and kind != EventKind.RUN_FINISHED:
                    record["payload"] = {"redacted": True}
                if capture == "metadata" and kind == EventKind.RUN_FINISHED:
                    record["payload"]["answer"] = None
                    record["payload"]["partial_answer"] = ""
                trace_file.write(json.dumps(record, allow_nan=False) + "\n")
                trace_file.flush()
            return event

        def observe(event: Observation) -> None:
            t = time.monotonic()
            try:
                active.observe(event)
            except Exception as exc:
                raise MethodError(f"{configured_spec.method_id}.observe: {type(exc).__name__}: {exc}") from exc
            finally:
                durations["controller"] += time.monotonic()-t

        def decide() -> Any:
            t = time.monotonic()
            try:
                return active.decide()
            except Exception as exc:
                raise MethodError(f"{configured_spec.method_id}.decide: {type(exc).__name__}: {exc}") from exc
            finally:
                durations["controller"] += time.monotonic()-t

        try:
            if trace_path is not None:
                path = Path(trace_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                trace_file = path.open("w", encoding="utf-8")
            if session is not None:
                threshold = session.acquire(run_id)
                active = dataclasses.replace(active, threshold=threshold, adaptive=True)  # type: ignore[type-var]
                self.preflight(self.backend, task, active, budget)
                configured_spec = active.spec
                config = method_configuration(active)
                context = dataclasses.replace(context, config_hash=stable_hash(config))
            try:
                active.reset(context)
            except Exception as exc:
                raise MethodError(f"{configured_spec.method_id}.reset: {exc}") from exc
            t = time.monotonic()
            handle = self.backend.open(context, task, ledger)
            durations["prefill"] += time.monotonic()-t
            event = emit(EventKind.RUN_STARTED, {"context": to_data(context),
                "task_hash": stable_hash(task.visible()), "method": to_data(configured_spec),
                "configuration": config, "capture": capture})
            yield event
            observe(event)
            phase = Phase.REASONING
            event = emit(EventKind.PHASE_CHANGED, {"phase": str(phase)})
            yield event
            observe(event)
            segmenter = Segmenter(configured_spec.boundary)
            decision = decide()
            finished = False
            while not finished:
                ledger.check()
                seen: set[str] = set()
                rounds = 0
                while not isinstance(decision, Continue):
                    rounds += 1
                    if rounds > max(16, budget.max_probes_per_prefix * 3):
                        raise MethodError("method did not exit decision cycle after terminal probe events")
                    if isinstance(decision, RequestSignals):
                        if not decision.requests:
                            raise MethodError("RequestSignals must contain at least one request")
                        phase = Phase.PROBING
                        failed_boundary = False
                        for request in decision.requests:
                            fingerprint = request.fingerprint()
                            usage = ledger.snapshot()
                            reserved_answer = budget.max_answer_tokens
                            available = min(budget.max_probe_output_tokens - usage.probe_output_tokens,
                                budget.max_total_generated_tokens - usage.total_generated_tokens - reserved_answer)
                            why = None
                            if request.prefix != handle.prefix:
                                why = "stale prefix"
                            elif fingerprint in seen:
                                why = "duplicate probe at unchanged decision state"
                            elif per_prefix.get(handle.prefix.identity, 0) >= budget.max_probes_per_prefix:
                                why = "per-prefix probe limit"
                            elif usage.probe_calls >= budget.max_probe_calls:
                                why = "per-run probe limit"
                            elif request.max_work.generated_tokens < 0 or request.max_work.generated_tokens > available:
                                why = "probe output reservation would consume reserved answer allowance"
                            if why:
                                event = emit(EventKind.PROBE_DENIED, ProbeFailure(request.request_id, why))
                                yield event
                                observe(event)
                                if fingerprint in seen:
                                    raise MethodError("method repeatedly requested the same probe after denial")
                                seen.add(fingerprint)
                                continue
                            seen.add(fingerprint)
                            per_prefix[handle.prefix.identity] = per_prefix.get(handle.prefix.identity, 0) + 1
                            event = emit(EventKind.PROBE_SCHEDULED, request)
                            yield event
                            observe(event)
                            scheduled_id = event.event_id
                            t = time.monotonic()
                            terminal_error: Cancelled | ReplayUnavailable | None = None
                            try:
                                ledger.begin_probe(request.max_work.generated_tokens,
                                                   request.max_work.scored_tokens, request.max_work.input_tokens)
                                probe_seed = probe_random_seed(context.probe_seed, request, ledger.probe_calls)
                                probe_seeds.append({"request_id": request.request_id, "seed": probe_seed})
                                result = handle.probe(request, probe_seed)
                                if result.prefix != handle.prefix or result.request_id != request.request_id:
                                    raise RuntimeError("backend returned stale or mismatched probe result")
                                if not result.valid:
                                    raise RuntimeError(result.error or "invalid probe result")
                                probes[request.request_id] = result
                                event = emit(EventKind.PROBE_COMPLETED, result, scheduled_id)
                            except (Cancelled, ReplayUnavailable) as exc:
                                terminal_error = exc
                                event = emit(EventKind.PROBE_FAILED, ProbeFailure(request.request_id,
                                             f"{type(exc).__name__}: {exc}"), scheduled_id)
                            except Exception as exc:
                                event = emit(EventKind.PROBE_FAILED, ProbeFailure(request.request_id,
                                             f"{type(exc).__name__}: {exc}"), scheduled_id)
                                failed_boundary = True
                            finally:
                                ledger.end_probe()
                                durations["probes"] += time.monotonic()-t
                            yield event
                            if terminal_error is not None:
                                raise terminal_error
                            observe(event)
                            if event.kind == EventKind.PROBE_FAILED and configured_spec.probe_failure_policy == "fail":
                                raise MethodError(f"probe failure policy=fail: {event.payload.error}")
                        phase = Phase.REASONING
                        # A failed observation is never evidence for stopping at this boundary.
                        decision = Continue() if failed_boundary else decide()
                    elif isinstance(decision, ReturnAnswer):
                        canonical = task.normalize(decision.answer)
                        source = probes.get(decision.provenance)
                        if canonical is None or source is None:
                            raise MethodError("ReturnAnswer requires a valid task answer and successful probe request provenance")
                        value = source.value
                        if isinstance(value, ParsedAnswer):
                            if not value.valid or not value.complete or canonical != value.answer:
                                raise MethodError("returned answer differs from validated probe answer")
                            selected_log_probs = value.token_log_probs
                        elif isinstance(value, CandidateScores):
                            if canonical not in value.candidates:
                                raise MethodError("answer missing from scored candidate set")
                        else:
                            raise MethodError("probe result cannot establish answer provenance")
                        ledger.accept_probe(source.operation_ids)
                        answer, status = canonical, RunStatus.COMPLETED
                        stop = StopInfo(decision.reason, last_event_id, ledger.snapshot().reasoning_tokens,
                                        decision.diagnostics, "returned_existing_answer")
                        finished = True
                        break
                    elif isinstance(decision, Abstain):
                        if not configured_spec.can_abstain:
                            raise MethodError("method has not declared abstention capability")
                        status = RunStatus.ABSTAINED
                        stop = StopInfo(decision.reason, last_event_id, ledger.snapshot().reasoning_tokens,
                                        finalization_outcome="abstained")
                        finished = True
                        break
                    elif isinstance(decision, Finalize):
                        stop = StopInfo(decision.reason, last_event_id, ledger.snapshot().reasoning_tokens,
                                        decision.diagnostics)
                        phase = Phase.ANSWERING
                        t = time.monotonic()
                        handle.transition(decision.recipe or configured_spec.finalization_recipe)
                        answer_text = getattr(handle, "answer_prefix", "")
                        event = emit(EventKind.PHASE_CHANGED, {"phase": str(phase),
                            "recipe": decision.recipe or configured_spec.finalization_recipe})
                        yield event
                        complete = False
                        while ledger.snapshot().answer_tokens < budget.max_answer_tokens:
                            ledger.check(generated_tokens=1)
                            token = handle.next_token(Phase.ANSWERING)
                            yield emit(EventKind.TOKEN_COMMITTED, token)
                            if token.eos:
                                complete = True
                                break
                            answer_text += token.text
                        durations["finalization"] += time.monotonic()-t
                        answer = task.normalize(answer_text) if complete else None
                        status = RunStatus.COMPLETED if complete and answer is not None else RunStatus.ANSWER_INCOMPLETE
                        selected_log_probs = tuple(getattr(handle, "answer_token_log_probs", ()))
                        stop = dataclasses.replace(stop, finalization_outcome=str(status))
                        finished = True
                        break
                    else:
                        raise MethodError(f"invalid decision {type(decision).__name__}")
                if finished:
                    break
                usage = ledger.snapshot()
                if (usage.reasoning_tokens >= budget.max_reasoning_tokens or
                    usage.total_generated_tokens >= budget.max_total_generated_tokens - budget.max_answer_tokens):
                    decision = Finalize("reasoning_budget")
                    continue
                phase = Phase.REASONING
                t = time.monotonic()
                ledger.check(generated_tokens=1)
                token = handle.next_token(Phase.REASONING)
                durations["reasoning"] += time.monotonic()-t
                event = emit(EventKind.TOKEN_COMMITTED, token)
                yield event
                if token.reasoning_end:
                    decision = Finalize("natural_reasoning_end")
                    continue
                if token.eos:
                    status = RunStatus.ANSWER_INCOMPLETE
                    stop = StopInfo("eos_before_answer", event.event_id, ledger.snapshot().reasoning_tokens,
                                    finalization_outcome="no_answer")
                    break
                observe(event)
                for boundary in segmenter.push(token.text):
                    event = emit(EventKind.STEP_BOUNDARY, boundary)
                    yield event
                    observe(event)
                decision = decide()
        except GeneratorExit:
            propagating = True
            ledger.cancellation_requests += 1
            if handle is not None:
                handle.cancel()
            raise
        except Cancelled as exc:
            status, error = RunStatus.CANCELLED, str(exc)
            ledger.cancellation_requests += 1
            if handle is not None:
                handle.cancel()
        except BudgetExceeded as exc:
            status, error = RunStatus.BUDGET_EXHAUSTED, str(exc)
        except MethodError as exc:
            status, error = RunStatus.METHOD_ERROR, str(exc)
        except ReplayUnavailable:
            propagating = True
            raise
        except Exception as exc:
            status, error = RunStatus.BACKEND_ERROR, f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if session is not None:
                    session_diagnostics = session.update(run_id, answer_log_probs=selected_log_probs,
                        output_tokens=ledger.snapshot().reasoning_tokens + ledger.snapshot().answer_tokens,
                        completed=status == RunStatus.COMPLETED and answer is not None)
            except Exception as exc:
                status, error = RunStatus.METHOD_ERROR, f"session update failed: {type(exc).__name__}: {exc}"
            finally:
                try:
                    if session is not None:
                        session.abort(run_id)
                finally:
                    try:
                        if handle is not None:
                            fallback_prefix = handle.prefix
                            handle.close()
                    except Exception as exc:
                        status, error = RunStatus.BACKEND_ERROR, f"backend cleanup failed: {exc}"
                    finally:
                        if trace_file is not None and propagating:
                            trace_file.close()
        phase = Phase.FINISHED
        durations["end_to_end"] = time.monotonic()-started
        stop = stop or StopInfo(str(status), last_event_id, ledger.snapshot().reasoning_tokens,
                                finalization_outcome=str(status))
        if status not in {RunStatus.COMPLETED, RunStatus.ABSTAINED}:
            stop = dataclasses.replace(stop, finalization_outcome=str(status))
        run_result = RunResult(run_id, answer, status, stop, ledger.snapshot(), durations,
            {"halt_version": "0.1.0rc1", **software, "method": to_data(configured_spec), "configuration": config,
             "backend": to_data(self.backend.info), "seed": seed, "probe_seed": context.probe_seed,
             "probe_invocation_seeds": probe_seeds, "probe_seed_policy": "settings_and_invocation_without_run_uuid_v1",
             "prompt_hash": stable_hash(task.render()), "task_hash": stable_hash(task.visible()),
             "budget": to_data(budget), "capture": capture, "session_initial": session_initial,
             "session_update": session_diagnostics,
             "timing_kind": "controller_only" if self.backend.info.name == "replay" else
                            "simulated" if self.backend.info.name == "scripted" else "inference"},
             answer_text, str(trace_path) if trace_path else None, error)
        try:
            yield emit(EventKind.RUN_FINISHED, run_result)
        finally:
            if trace_file is not None:
                trace_file.close()
