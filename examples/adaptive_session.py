"""Explicit ordered adaptive session using simulated answers and embeddings."""
from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.methods import Refrain
from halt.session import RefrainSession

session = RefrainSession(seed=7)
runner = HaltRunner(ScriptedBackend(reasoning=[
    "The answer is C.\n\n", "Wait, let me check the answer.\n\n",
]))
task = MultipleChoiceTask("12 minus 5?", {"A": "5", "C": "7"})
for request_id in ("first", "second", "third"):
    result = runner.run(task, Refrain(), Budget(), session=session, run_id=request_id)
    print(result.answer, result.provenance["session_update"])
session.save("runs/refrain_session.json")
