# Try your dataset

HALT reads local CSV and JSONL files. There is no upload service: point the command
at your file, and results are saved to a local directory. Multiple-choice and
numeric-answer questions are supported.

From a source checkout, install HALT and the real-model dependencies:

```sh
python -m pip install -e ".[transformers]"
```

Check your file before starting an experiment:

```sh
halt dataset check questions.csv
halt benchmark --dataset questions.csv --methods full_reasoning,halt_cot --output-dir runs/my_questions
```

The second command uses the tested, pinned Qwen3-0.6B model by default. Its weights
are downloaded if they are not cached. The terminal shows a comparison table;
`runs/my_questions` contains `report.html`, `results.csv`, `results.jsonl`,
`summary.csv`, and `manifest.json` for your own analysis. The selected model,
settings, dataset identity, and import mappings are recorded.

To check the complete command workflow without model dependencies or downloads:

```sh
python -m pip install -e .
halt benchmark --dataset data/example_questions.csv --backend scripted --methods full_reasoning,immediate_answer --output-dir runs/dataset_demo
```

Scripted results use predetermined outputs. They demonstrate the workflow and do
not measure a language model's quality or efficiency.

## CSV

Use a header and one question per record. This file is ready to run:

```csv
id,question,A,B,C,D,answer
addition,What is 2 + 2?,3,4,5,6,B
subtraction,What is 9 - 4?,5,6,7,8,A
```

The `answer` column contains the correct **choice label**. The visible choices
come exclusively from the choice columns. HALT never constructs options from the
answer column. Labels `A` through `Z`, `choice_A`, and `option_A` are recognized
automatically, including lowercase variants. Each selected choice must contain
nonempty text; use an explicit choices object when questions have varying numbers
of options.

Normal CSV quoting works for questions containing commas, quotes, or newlines.
UTF-8 files with or without a byte-order mark are accepted. Excel users can save
as **CSV UTF-8**; `.xlsx` files are not read directly.

## JSONL

Each nonempty line must be a complete JSON object:

```jsonl
{"id":"addition","question":"What is 2 + 2?","choices":{"A":"3","B":"4"},"answer":"B"}
{"id":"subtraction","question":"What is 9 - 4?","choices":["5","6","7"],"answer":"A"}
```

A choices object preserves its explicit labels and order. A choices list receives
labels `A`, `B`, `C`, and so on, up to 26 options. List answers must use these
labels; zero-based numeric indexes are not guessed. Options must be strings,
including numbers used as option text. CSV can also store the same choices object
or list as JSON in a `choices` column, using normal CSV escaping.

## Numeric questions

Omit choices and use numeric answers:

```csv
id,question,answer
addition,What is 2 + 2?,4
division,What is 3 divided by 2?,1.5
```

```sh
halt benchmark --dataset numeric_questions.csv --methods full_reasoning,answer_convergence --output-dir runs/numeric_questions
```

With `--adapter auto`, choices indicate multiple-choice questions; without them,
answers must satisfy the numeric answer contract. Numeric normalization supports
signed decimals, scientific notation, and correctly grouped thousands separators.
It compares exact normalized values, without a tolerance. JSON decimal references
retain their precision. Fractions such as `1/2`, units such as `4 kg`, essays, and
code answers need a separate task/evaluation extension. A file must use one task
kind; HALT does not silently switch an individual row to another grading rule.

Methods requiring answer candidates cannot run on numeric tasks. Invalid
method/task combinations are checked before model loading. Use `--adapter mcq`
or `--adapter numeric` when you want to declare the task kind explicitly.

## Column detection and mappings

The friendly `auto`, `mcq`, and `numeric` adapters recognize these common names,
ignoring letter case and surrounding spaces:

| Field | Recognized names |
|---|---|
| Question | `question`, `prompt`, `query`, `problem` |
| Answer | `answer`, `answer_key`, `answerKey`, `correct_answer`, `label`, `target`, `gold` |
| ID | `id`, `question_id`, `item_id`, `example_id`, `uid` |
| Choices JSON | `choices`, `options` |

Spaces and hyphens in these names are treated like underscores. If multiple names
could represent the same field, HALT asks for a mapping instead of guessing. For
example, for columns `key,body,expected,left,right`:

```sh
halt dataset check custom.csv --question-column body --answer-column expected --id-column key --choice-columns A=left,B=right
halt benchmark --dataset custom.csv --question-column body --answer-column expected --id-column key --choice-columns A=left,B=right --methods full_reasoning,halt_cot
```

Use `--choices-column alternatives` for a single column containing JSON choices.
Use either `--choices-column` or `--choice-columns`. An explicit choices mapping
also resolves ambiguity when both JSON choices and separate option columns exist.
Question, answer, ID, and selected choice fields must use distinct source columns.

If the file has no ID column, HALT generates stable IDs from source line numbers,
such as `row-2` for the first ordinary CSV data row. If an ID column exists, every
record needs a nonempty, unique ID. Explicit IDs are useful when comparing
selections or revised files.

Unselected columns such as category or author are ignored by inference and grading.
Their names are listed by `halt dataset check questions.csv --json` and recorded
in the experiment's import metadata. They are not copied into per-question result
rows; use the question ID to join your original metadata when doing custom analysis.

## Validation and repeatability

```sh
halt dataset check questions.csv --json
halt benchmark --dataset questions.csv --limit 10 --seed 19
```

The checker reports source row numbers and collects recoverable row errors,
including missing values, invalid answers, duplicate IDs, malformed JSONL, and
incorrect CSV field counts. A broken CSV quoting sequence can prevent parsing the
remainder of that file and is reported as a file parsing failure. No partial
dataset is returned when validation fails.

`--limit 10` selects the first ten questions **after validating the entire file**.
A bad later row must still be fixed. Dataset identity includes the complete file
hash, selected ID order, task adapter, and detected or explicit import mapping.
Advanced configurations can also declare `evaluation.split` and
`evaluation.data_revision`. Keep development/calibration and final test questions
separate when choosing thresholds.

Existing `mcq_jsonl` and `numeric_jsonl` adapters retain their strict field schema:
`id`, `question`, `answer`, and `choices` for MCQ. They reject unknown fields and
column mappings. Choose `auto`, `mcq`, or `numeric` for CSV, common-column
detection, extra metadata columns, or generated IDs.

## Python interface

```python
from halt.evaluation.datasets import load_dataset, validate_dataset

report = validate_dataset("questions.csv", adapter="auto", limit=10)
if not report.valid:
    raise ValueError(report.to_dict()["errors"])

dataset = report.dataset
# Or load directly and receive ConfigurationError if any row is invalid:
dataset = load_dataset("questions.csv", adapter="auto", limit=10)
```

Both functions accept `question_column`, `answer_column`, `id_column`,
`choices_column`, and `choice_columns={"A": "left", "B": "right"}`. The validation
report contains detected columns, ignored columns, counts, structured errors, and
the dataset only when all rows are valid. Importing or validating a dataset does
not load a model. Gold references stay in evaluation items; the runtime receives
only the question and its visible choices.
