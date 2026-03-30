# Changelog

## Unreleased

### ND2 Spectral Export

- Added recursive batch OME-Zarr scanning and loading from a parent folder.
- Improved the batch Zarr table so all rows use a stable readable background instead of bright alternating rows.
- Fixed text visibility issues in the batch converter/load tables.

### Floating Dock Windows

- Improved floating dock behavior so plugin windows are easier to move and resize as standalone Qt dock windows.

### Truecolor Rendering

- Updated the internal truecolor rendering gamma from `0.9` to `1.4`.
- Kept truecolor generation in the plugin separate from napari's layer gamma setting.

### Spectral Viewer

- Reorganized the UI into 3 sections:
  - `ROI Spectrum`
  - `ROI Comparison`
  - `Pseudocolor`
- Removed the redundant active-image ROI prepare action from the main workflow and kept a single `Prepare ROI Layers` action.
- Added an `ROI image` selector and `Activate ROI Layer` action for faster navigation across many open images.
- Made ROI helper visibility follow the active image context so unrelated ROI overlays hide automatically.
- Reordered ROI layers to stay adjacent to their source spectral image in the napari layer list.
- Moved ROI annotation text onto the Shapes layer so no separate visible annotation layer row is created.
- Added a multi-image ROI comparison table with `Plot Selected Across Images`.
- Made `Normalized/Absolute`, individual ROI plotting, pooled ROI plotting, and related controls reactively redraw the active plot.
- Improved plot layout with larger default plot height, unclipped wavelength labels, and optional outside legends.
- Added session packaging:
  - `Save Session Package`
  - `Load Session Package`
- Session packages now save:
  - `manifest.json`
  - ROI shape JSON files
  - ROI dataset JSON files
  - truecolor TIFF files

### ROI Dataset Storage

- Added shared-store change notifications so widgets can refresh automatically when ROI datasets are added, updated, or removed.
- Fixed ROI storage so raw spectra are preserved in memory and normalization is applied only at display time.

### Spectral Analysis

- Fixed the stored dataset table row styling to avoid bright alternating rows.
- Connected the analysis widget to the shared ROI store so dataset tables refresh automatically when Spectral Viewer updates ROI datasets.
- Added a dedicated `Stats` section for:
  - descriptive statistics
  - normality and equality-of-variance checks
  - two-group Welch t-test
  - correlation analysis
- Added a larger report-style statistics view instead of relying on a compact status line.
- Added `Export Stats Report` for saving the statistical report as text or CSV.
- Generalized the t-test so it can operate on any selected factor with exactly two valid groups, not only WT/HNPP-style labels.
- Increased the analysis plot area and improved plot spacing so report plots are easier to review.

### Notes

- Session-package reload still depends on access to the original spectral source paths for full image-layer restoration.
- Saved truecolor TIFFs are included in the package as derived outputs, but they do not replace the original spectral source data.
