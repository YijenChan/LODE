# LODE

LODE reconstructs long-range advanced persistent threat (APT) scenarios from a fixed audit-log interval. It avoids materializing a full-history provenance graph: a process-conditioned GRU selects investigation seeds, source/destination postings retain complete event history, and a bounded LLM investigator retrieves witnessed evidence before producing direct-node judgments and an APT summary graph.

## Method overview

1. **Recurrent Event Anomaly Detection.** A shared GRU predicts the next typed signature `(relation, peer entity type, process role)` in each responsible-process stream. Negative log-likelihood scores are aggregated by process window and calibrated against benign validation windows.
2. **Indexed APT Evidence Space.** Every canonical event is retained in disk-backed source and destination postings. An anchored query retrieves bounded `past/future` evidence in `earliest/near` order without constructing a NetworkX/DGL full-history graph.
3. **Evidence-Grounded Reconstruction.** The investigator selects only visible event IDs, attaches incident evidence to every direct-node claim, and retrieves additional history through retained anchors. Deterministic checks reject unsupported output. The committed graph contains claim-relevant witnessed events and their seed connectors.

## Repository layout

```text
lode/detector.py       process-conditioned GRU and benign calibration
lode/archive.py        retain-all source/destination postings and retrieval
lode/investigator.py   bounded LLM loop, validation, and graph export
lode/commit.py         claim-support and connector commitment
configs/               example experiment configuration
docs/                  data contract and methodology trace
tests/                 unit tests that require no dataset or API call
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Input

LODE expects chronological canonical Parquet files under
`canonical/{train,validation,test}/sorted/`. See
[docs/data-format.md](docs/data-format.md) for the required columns. The
trajectory-selection protocol is supplied separately under `protocol/`; it
defines the fixed audit-log interval but is never read as a detection label.

Copy `configs/e3_theia.yaml` and update `data_root`, `output_root`,
`archive_root`, and `protocol`. The relation vocabulary is declared explicitly
so all splits use the same event schema.

## Running LODE

From the repository root:

```bash
# Train the benign GRU, score validation/test streams, and create seed anchors.
python -m lode.detector --config configs/e3_theia.yaml all

# Build the retain-all numeric postings for the investigation split.
python -m lode.archive --config configs/e3_theia.yaml

# Run bounded reconstruction and export the APT summary graph.
set OPENAI_API_KEY=YOUR_KEY       # Windows cmd
# $env:OPENAI_API_KEY="YOUR_KEY" # Windows PowerShell
# export OPENAI_API_KEY=YOUR_KEY  # Linux/macOS
python -m lode.investigator --config configs/e3_theia.yaml
```

The final directory `runs/<run_id>/summary_graph/` contains:

- `direct_nodes.jsonl`: validated directly malicious node judgments;
- `nodes.jsonl`: nodes retained in the attack summary graph;
- `edges.jsonl`: witnessed events with original direction, timestamp, and raw reference;
- `cases.json`: per-seed evidence, claims, calls, and commitment audit.

No test label is loaded by these stages. Evaluation should be performed after
the outputs are frozen, using a separate scorer and the same registered
trajectory protocol for all methods.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover typed signatures, next-event loss, window aggregation, anchored
retrieval, response validation, and graph commitment. They do not send API
requests.

## Reproducibility notes

- Random seeds and detector hyperparameters are explicit in the YAML config.
- Event row positions remain aligned across scoring, retrieval, and raw-reference export.
- Archived events are never modified by LLM responses.
- Summary-graph commitment guarantees weak seed connectivity only; directed, time-respecting connectivity is evaluated separately.
- API credentials are read only from `OPENAI_API_KEY` and excluded by `.gitignore`.
