# Scientific Figure Draft Set

This folder stores manually editable scientific figure drafts for the CCHP paper.

## Structure

- `svg/`
  - vector figure drafts for visual ideation and later refinement
- `pdf/`
  - export-ready versions that can be included directly in LaTeX
- `tex/`
  - LaTeX insertion snippets with suggested captions and labels

## Figure List

- `fig_intro_problem_overview`
  - introduction-level paper overview
- `fig_cchp_simplified_layout`
  - Section 2 plant layout and coupled energy pathways
- `fig_physics_aware_drl_pipeline`
  - Section 3 control and safety-projection pipeline
- `fig_dynamic_gt_om_allocation`
  - Section 3 dynamic GT O&M allocation comparison

## Notes

- The SVGs are intended as research-figure drafts, not final submission artwork.
- The matching `tex/` files now point to the `pdf/` exports for a safer default LaTeX workflow.
- Edit the SVGs first, then re-export the corresponding PDFs when you refine typography or spacing.
