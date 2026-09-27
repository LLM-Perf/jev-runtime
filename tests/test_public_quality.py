from benchmarks.prepare_public_quality import select


def test_public_sampling_is_balanced_group_disjoint_and_reproducible():
    source = {
        "fit": [
            {"text": f"training example {label} {index}", "label": label}
            for label in range(2)
            for index in range(6)
        ],
        "heldout": [
            {"text": f"heldout example {label} {index}", "label": label}
            for label in range(2)
            for index in range(6)
        ],
    }
    source["fit"] += [{"text": " A COMMON TEXT ", "label": 0}]
    source["heldout"] += [{"text": "a common text", "label": 0}]
    source["fit"] += [{"text": "conflict", "label": 0}, {"text": "conflict", "label": 1}]
    chosen, accounting = select(source, 3, 2)
    assert accounting["cross_split_groups_excluded_from_fit"] == 1
    assert accounting["conflicting_groups_excluded"] == 1
    for rows in chosen.values():
        assert len(rows) == 6 and sum(row["label"] == 0 for row in rows) == 3
    assert not (
        {row["group_id"] for row in chosen["fit"]} & {row["group_id"] for row in chosen["heldout"]}
    )
    assert chosen == select(source, 3, 2)[0]
