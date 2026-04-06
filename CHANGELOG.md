# Changelog

## 1.4.0 - 2026-04-05

### Spatial Ratio Analysis

- Added a new `Spatial Ratio Analysis` dock widget to the plugin manifest so users can launch kernel-based spatial-ratio workflows directly from the napari Plugins menu.
- Added per-image ROI context tracking for shape ROIs, bound labels layers, and combined shape-plus-label refinement so users can see which images are ready for kernel analysis.
- Added kernel-based spatial ratio computation with configurable kernel size, minimum valid coverage, split wavelength, ratio mode, optional normalization, and optional blank-reference subtraction.
- Added scatter-selection, raw-kernel CSV export, summary CSV export, and `Send Summary To Analysis` so kernel results can be filtered, exported, and handed off into downstream statistical analysis.

### Spectral Viewer

- Improved ROI handling so bound labels can refine shape ROIs instead of forcing a labels-only workflow, and labels-only analysis still works when no shapes are present.
- Added optional blank-reference subtraction during ROI spectrum measurement.
- Expanded the ROI curation table with analysis toggles and editable metadata fields including `group_label`, `animal_id`, `sex`, `age`, `region`, and `roi_class`.
- Updated ROI status messaging and context refresh behavior so image, ROI, and labels state stays synchronized while users prepare and curate datasets.

### ROI Dataset Storage

- Expanded stored ROI dataset metadata to include trace kind/label, measurement kind, analysis level, acceptance flags, ROI class, kernel settings, spatial-ratio summary metrics, and mean intensity.
- Updated dataset replacement and CSV export logic so spectral and spatial-ratio datasets preserve the richer metadata needed for curation and downstream analysis.

### Spectral Analysis

- Added a measurement filter so the analysis widget can switch between `spectral_mean` and `spatial_ratio` datasets.
- Added `trace_kind`, `trace_label`, and `roi_class` to the dataset table and extended grouping/statistical factor options to include `roi_class`.
- Switched analysis selection to persistent per-dataset `use_for_analysis` flags stored in the shared ROI dataset store.

## 1.3.1 - 2026-04-05

### README

- Added an interface preview screenshot for the `ND2 Spectral Export` workflow.
- Documented the highlighted conversion, Zarr browsing, and dataset-property table regions in the screenshot.

### ND2 Spectral Export

- Increased the default floating height of the `ND2 Spectral Export` window.
- Fixed the export layout so the bottom controls stay anchored while the dataset table expands to show more rows.

## 1.3.0 - 2026-04-05

### ND2 Spectral Export

- Added recursive batch OME-Zarr scanning and loading from a parent folder.
- Improved the batch Zarr table so all rows use a stable readable background instead of bright alternating rows.
- Fixed text visibility issues in the batch converter/load tables.
- Reworked `ND2 Spectral Export` into a drag-and-drop workflow with dedicated source and output boxes plus a single `Convert To OME-Zarr` action.
- Added recursive ND2 conversion from a dropped folder, subfolder, or single `.nd2` file.
- Preserved the original relative folder structure under the chosen output root instead of flattening converted datasets into one folder.
- Added export `manifest.json` saving at the output root, following the safer temp-write-and-replace manifest pattern used in workspace persistence.
- Kept per-dataset export metadata in OME-Zarr attributes and also recorded source/output paths plus metadata in the root export manifest.
- Merged OME-Zarr scanning and opening into the same export section so users can browse, scan, and open `.zarr` datasets from the main conversion workspace.
- Reused the main table for ND2 conversion status so users can see queued files, converted outputs, and per-file failures in one place.
- Changed conversion behavior so single-file failures are skipped and logged while the remaining ND2 files continue converting.
- Added a conversion progress bar plus a dedicated `Conversion Errors` panel for troubleshooting failed files.
- Added more prominent primary-action styling for `Scan Zarr Folder` and `Convert To OME-Zarr`.
- Improved drop-target styling with visible in-box prompts such as `Drop ND2 file or folder here` so the drag-and-drop workflow is more obvious.
- Tightened the export layout so browse actions sit beside each drop target and the converter no longer wastes a full row for each browse button.
- Moved the main conversion status, progress bar, and convert action together at the top of the export section while keeping conversion errors isolated at the bottom.
- Fixed export-window resizing so long status messages no longer change the dock width or height during conversion.
- Added a fixed-height `Status:` message bar with elided text and tooltips for full messages.
- Increased the shared table height so queued ND2 files and scanned Zarr datasets are easier to see.
- Added grouped row coloring by `relative_path` in Zarr scan results to make folder group boundaries easier to spot.
- Added multi-row selection for Zarr scan results, including row-based `Open Selected Zarr` behavior and space-bar toggling of selected rows.
- Fixed drag-and-drop repaint behavior so the dragged folder icon does not remain visible as a ghost after dropping.

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
- Added `Prepare Selected ROI` so users can prepare ROI context for only the chosen image instead of walking every open spectral layer.
- Kept `Prepare ROI Layers` for multi-image setup, but stopped it from stealing the active selection while preparing all images.
- Updated the `ROI image` controls so users can jump back to the source image with `Find Image` instead of forcing ROI-layer activation.
- Made ROI helper visibility follow the active image context so unrelated ROI overlays hide automatically.
- Reordered ROI layers to stay adjacent to their source spectral image in the napari layer list.
- Moved ROI annotation text onto the Shapes layer so no separate visible annotation layer row is created.
- Added a multi-image ROI comparison table with `Plot Selected Across Images`.
- Added `Remove Selected Rows` to the ROI comparison table so unneeded ROI or pooled traces can be deleted from stored datasets.
- Made `Normalized/Absolute`, individual ROI plotting, pooled ROI plotting, and related controls reactively redraw the active plot.
- Triggered ROI spectrum refresh when users move between source and ROI-context layers so the active image workflow updates more naturally.
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
