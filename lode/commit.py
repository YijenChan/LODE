"""Commit witnessed claim support and BFS connectors to an APT summary graph."""
from collections import defaultdict, deque


def commit_case(events: list[dict], direct_nodes: list[int], seed_id: int) -> tuple[list[dict], dict]:
    """Return only direct-claim incident events and their seed connectors.

    Connectivity is computed on the undirected projection. Original direction,
    timestamp, and raw-event identity remain attached to every committed event.
    """
    by_id = {event["id"]: event for event in events}
    malicious = {int(node) for node in direct_nodes}
    protected = {
        event["id"] for event in events
        if event["src"] in malicious or event["dst"] in malicious
    }
    if not protected:
        return [], {"reason": "no supported direct-node claim", "protected": 0, "connectors": 0}
    if seed_id not in by_id:
        return [], {"reason": "seed absent from selected evidence", "protected": len(protected), "connectors": 0}

    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        adjacency[event["src"]].append((event["dst"], event["id"]))
        adjacency[event["dst"]].append((event["src"], event["id"]))

    root = by_id[seed_id]["src"]
    predecessor: dict[int, tuple[int, int] | None] = {root: None}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for other, event_id in sorted(adjacency[node], key=lambda item: item[1]):
            if other not in predecessor:
                predecessor[other] = (node, event_id)
                queue.append(other)

    keep = {seed_id}
    for event_id in sorted(protected):
        event = by_id[event_id]
        if event["src"] not in predecessor or event["dst"] not in predecessor:
            continue
        keep.add(event_id)
        for endpoint in (event["src"], event["dst"]):
            node = endpoint
            while predecessor[node] is not None:
                node, connector = predecessor[node]
                keep.add(connector)

    committed = [event for event in events if event["id"] in keep]
    return committed, {
        "reason": "claim support plus seed connectors",
        "protected": len(protected),
        "connectors": len(keep - protected),
        "removed": len(events) - len(committed),
        "directed_path_claim": False,
    }
