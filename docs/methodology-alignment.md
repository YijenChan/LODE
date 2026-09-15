# Methodology-to-code alignment

| Paper component | Implementation | Contract |
| --- | --- | --- |
| Recurrent Event Anomaly Detection | `lode/detector.py` | A shared GRU predicts `(relation, peer type, process role)` in each responsible-process stream. Events are scored before state update. Process-window maxima are calibrated by the benign validation empirical CDF. |
| Indexed APT Evidence Space | `lode/archive.py` | Every canonical test event is retained through numeric source/destination postings. Queries are anchored by an observed entity and event position and support `past/future` with `earliest/near` ordering. |
| Evidence-Grounded Reconstruction | `lode/investigator.py` | The investigator can select only visible event IDs, attach incident support to direct-node claims, and query through retained anchors under a fixed context budget. A valid response replaces the previous judgment state. |
| Attack summary graph | `lode/commit.py` | The output contains witnessed claim-incident events and BFS connectors in the selected undirected component. Original edge direction, timestamp, and raw-event reference are preserved. Directed temporal connectivity is evaluated separately. |

The detector, archive, and investigator do not import attack labels. Dataset labels and baseline implementations are intentionally outside this repository.

