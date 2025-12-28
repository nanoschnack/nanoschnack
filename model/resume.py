"""Helpers for persisting dataset resume offsets across runs."""


def normalize_resume_rows(resume_state, dataset_specs):
    # Collect per-spec row offsets from a checkpoint and keep any unknown specs.
    resume_rows = {}
    if isinstance(resume_state, dict):
        for entry in resume_state.get("datasets", []):
            spec_key = entry.get("spec")
            if spec_key:
                resume_rows[spec_key] = int(entry.get("row_offset", 0) or 0)

    # Ensure current dataset specs are present with default offsets.
    for spec in dataset_specs:
        resume_rows.setdefault(spec["spec"], 0)
    return resume_rows


def cap_resume_rows(resume_rows, total_rows_by_spec):
    # Clamp resume offsets to known totals and align full passes across specs.
    if not total_rows_by_spec:
        return dict(resume_rows)

    adjusted = dict(resume_rows)
    totals = {
        spec_key: total_rows_by_spec.get(spec_key)
        for spec_key in adjusted.keys()
    }
    valid_totals = {spec_key: total for spec_key, total in totals.items() if total}
    if not valid_totals:
        return adjusted

    pass_counts = [
        adjusted.get(spec_key, 0) // total
        for spec_key, total in valid_totals.items()
    ]
    min_passes = min(pass_counts) if pass_counts else 0
    if min_passes:
        for spec_key, total in valid_totals.items():
            adjusted[spec_key] = max(0, adjusted.get(spec_key, 0) - (min_passes * total))

    for spec_key, total in valid_totals.items():
        if adjusted.get(spec_key, 0) > total:
            adjusted[spec_key] = total
    return adjusted


def build_resume_state(source_row_counts, dataset_specs):
    # Emit current specs first so checkpoints remain easy to read.
    datasets = []
    seen = set()
    for spec in dataset_specs:
        spec_key = spec["spec"]
        datasets.append(
            {
                "spec": spec_key,
                "row_offset": source_row_counts.get(spec_key, 0),
            }
        )
        seen.add(spec_key)

    # Preserve offsets for specs not in the current run.
    for spec_key in sorted(key for key in source_row_counts.keys() if key not in seen):
        datasets.append(
            {
                "spec": spec_key,
                "row_offset": source_row_counts.get(spec_key, 0),
            }
        )
    return {"datasets": datasets}


def is_resume_exhausted(row_offset, total_rows):
    return total_rows is not None and row_offset >= total_rows
