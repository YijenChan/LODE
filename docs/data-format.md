# Canonical input format

Each split lives under `canonical/<split>/sorted/` and contains two Parquet files sorted by `timestamp_ns`.

`events.parquet` requires:

`event_id`, `timestamp_ns`, `src_uuid`, `src_type`, `relation`, `dst_uuid`, `dst_type`, `source_file`, and `source_line`.

`nodes.parquet` requires:

`node_uuid`, `node_type`, `attributes_json`, `source_file`, and `source_line`.

Stable row positions in `events.parquet` are used as internal event IDs. The event schema in the run configuration must cover every relation and entity type. No ground-truth label is part of this interface.
