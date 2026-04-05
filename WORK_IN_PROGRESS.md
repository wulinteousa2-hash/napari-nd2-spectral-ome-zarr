# Work In Progress

## 2026-03-22

- Expanded `ND2 Spectral Export` into a unified ND2/Zarr workspace instead of a separate single-file OME-Zarr loader
- Added drag-and-drop Zarr source scanning and batch opening from the same export widget
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
- Added root-level export `manifest.json` saving plus conversion progress and failure reporting for ND2 batch export

## Recent ROI / Analysis work

- Added per-image ROI layers so ROI numbering resets by image
- Moved ROI annotation text onto the Shapes layer so no separate visible annotation row is needed
- Added in-memory stored ROI datasets for later analysis
- Added CSV export for stored ROI datasets
- Added `Spectral Analysis` as a third napari widget
- Added dataset metadata editing, dataset inclusion checkboxes, and dataset removal actions
- Added ROI, image, and animal summary tables
- Added split-wavelength ratio analysis and group comparison workflow
- Added ROI comparison row removal plus selected-image ROI preparation and source-image navigation
