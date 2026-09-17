# Methodology-to-code alignment

| Paper component | Implementation | Contract |
| --- | --- | --- |
| Recurrent Event Anomaly Detection | `lode/detector.py` | A shared GRU predicts `(relation, peer type, process role)` in each responsible-process stream. Events are scored before state update. Process-window maxima are calibrated by the benign validation empirical CDF. |
| Indexed APT Evidence Space | `lode/archive.py` | Every canonical test event is retained through numeric source/destination postings. Queries are anchored by an observed entity and event position and support `past/future` with `earliest/near` ordering. |
| Evidence-Grounded Reconstruction | `lode/investigator.py` | The investigator can select only visible event IDs, attach incident support to direct-node claims, and query through retained anchors under a fixed token and retrieval-page budget. A valid response replaces the previous judgment state. |
| Attack summary graph | `lode/commit.py` | The output contains witnessed claim-incident events and BFS connectors in the selected undirected component. Original edge direction, timestamp, and raw-event reference are preserved. Directed temporal connectivity is evaluated separately. |

The detector, archive, and investigator do not import attack labels. Dataset labels and baseline implementations are intentionally outside this repository.

## Reported implementation profile

| Manuscript setting | Configuration or enforcement |
| --- | --- |
| One-layer GRU; 32-D embedding; 64 hidden units | `pids.layers`, `pids.embedding_dim`, and `pids.hidden_dim`; the detector rejects a non-one-layer profile. |
| Four Adam epochs at $10^{-3}$ | `pids.epochs: 4` and `pids.learning_rate: 0.001`. |
| 15-minute process-score aggregation | `pids.window_seconds: 900`; recurrent states still follow complete process sequences. |
| Up to eight retrieval pages per seed | `retrieval.pages_per_seed: 8`; the initial packet is page 1 and each anchored follow-up consumes at most one additional page. |
| At most 4,096 input tokens per reasoning call | `investigator.max_prompt_tokens: 4096`; `lode/investigator.py` tokenizes the serialized message array with `o200k_base` and trims the current packet before the request. |
| Fixed prompt and temperature 0 | `SYSTEM_PROMPT` in `lode/investigator.py` and `investigator.temperature: 0.0`; both are applied to every request. |
| Three end-to-end runs | `experiment_seeds: [7, 19, 43]`; one config/run directory is used per active `seed`. |

The repository implements LODE itself. Released baseline settings, baseline
code, datasets, reference annotations, and reported result tables remain
outside this repository, as stated in the README.
