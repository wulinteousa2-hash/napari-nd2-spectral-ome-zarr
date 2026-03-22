# Work In Progress

## 2026-03-22

- Added a dedicated single-file OME-Zarr loader to `ND2 Spectral Export`
- Added drag-and-drop Zarr input with dataset inspection before opening
- Added view selection for Zarr loading:
  - `Visible sum`
  - `Truecolor`
  - `Raw spectral`
- Added preview-level toggle for Zarr-derived display layers
- Added recursive batch Zarr scanning from a parent folder
- Added batch Zarr table with per-dataset open checkbox and metadata preview
- Changed batch Zarr `relative_path` to show only the parent folder, such as `./659/`
- Changed `Visible sum` to use raw per-pixel mean across spectral bins instead of normalized sum
- Made `ND2 Spectral Export`, `Spectral Viewer`, and `Spectral Analysis` float by default

## Recent ROI / Analysis work

- Added per-image ROI layers so ROI numbering resets by image
- Replaced risky Shapes text labels with a separate ROI label layer
- Added in-memory stored ROI datasets for later analysis
- Added CSV export for stored ROI datasets
- Added `Spectral Analysis` as a third napari widget
- Added dataset metadata editing, dataset inclusion checkboxes, and dataset removal actions
- Added ROI, image, and animal summary tables
- Added split-wavelength ratio analysis and group comparison workflow
