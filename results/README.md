# Results Archive

This folder is the structured landing zone for exported experiment results, paper-ready tables, and generated figures.

## Layout

- `paper/`
  - human-readable markdown snapshots and paper-facing result notes
- `tables/`
  - machine-readable CSV or JSON exports derived from finalized runs
- `figures/`
  - rendered figure assets grouped by topic and format

## Current Convention

- Submission-facing normalized archive:
  - `results/archive/paper_ready_<date>/`
- Same-info DRL paper summaries:
  - `results/paper/drl_with_mlp/snapshots/`
- Latest Kaggle MLP figure exports:
  - `results/figures/paper/latest_kaggle_mlp/{png,pdf,eps,tiff}/`
- Latest Kaggle MLP tabular exports:
  - `results/tables/paper/latest_kaggle_mlp/`

## Submission Rule

- `results/` is the only folder that should be treated as submission-facing output.
- Raw experiment directories under `runs/`, `kaggle/`, and `tmp_verify/` remain source locations.
- When a run becomes paper-relevant, copy its compact artifacts into `results/archive/paper_ready_<date>/`.
- Plotting scripts should prefer `results/tables/` and `results/archive/` over direct reads from `runs/` or `kaggle/`.

## Notes

- `project_document/paper/` remains the writing workspace.
- `results/` is the publication-oriented archive for stable outputs.
- Plotting scripts should default to writing here instead of under `scripts/plot/`.
