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
