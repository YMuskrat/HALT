"""No-network example: compare three baseline and research controllers."""
from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.methods import AnswerConvergence, HaltCoT, ThinkBrake

task = MultipleChoiceTask("12 minus 5?", {"A": "5", "B": "6", "C": "7", "D": "8"})
runner = HaltRunner(ScriptedBackend())
for method in (HaltCoT(), ThinkBrake(), AnswerConvergence(max_probe_tokens=8)):
    result = runner.run(task, method, Budget())
    print(method.spec.method_id, result.answer, result.status, result.stop.reason)
