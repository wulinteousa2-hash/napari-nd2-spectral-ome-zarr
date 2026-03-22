# napari-nd2-spectral-ome-zarr

napari plugin for Nikon ND2 spectral imaging workflows with emphasis on 2D spectral cubes.

Features:

- Read `.nd2` files into napari
- Read `.zarr` and `.ome.zarr` through the plugin loader widget
- Build an estimated truecolor RGB view for spectral ranges spanning roughly 400 nm to 740 nm
- Export the loaded ND2 spectral cube to OME-Zarr with multiscales and wavelength metadata
- Plot ROI spectra from spectral OME-Zarr layers
- Keep per-image ROI spectral datasets in memory during the napari session
- Export stored ROI datasets and analysis tables to CSV
- Run split-wavelength Nile Red ratio analysis, aggregation, and group comparison in a dedicated analysis panel

The plugin is designed around 2D spectral images and keeps `T`, `C`, `Z`, `Y`, `X` axis semantics explicit during export.

## Dock Widgets

The plugin now exposes 3 napari dock widgets:

- `ND2 Spectral Export`
- `Spectral Viewer`
- `Spectral Analysis`

All 3 widgets are configured to float by default instead of staying docked in the main napari window.

## ND2 Spectral Export Workflow

`ND2 Spectral Export` now handles:

- single ND2 preview
- batch ND2 to OME-Zarr export
- single OME-Zarr loading
- batch OME-Zarr loading from a parent folder

### Single Zarr loading

The widget supports drag-and-drop or browsing for one `.zarr` folder.

Before opening, it shows:

- dataset name
- axes
- full-resolution shape
- preview shape
- whether the dataset is spectral
- wavelength count and wavelength range

The user can choose which views to open:

- `Visible sum`
- `Truecolor`
- `Raw spectral`

It also supports `Use preview pyramid level` for the display layers.

### Batch Zarr loading

The widget also supports scanning one parent folder recursively for `.zarr` datasets, including nested subfolders.

The batch table shows:

- `open`
- `name`
- `relative_path`
- `axes`
- `shape`
- `preview_shape`
- `wavelengths`
- `spectral`

`relative_path` is shown as the parent folder only, for example `./659/`, instead of including the `.zarr` folder name.

The user can:

- browse to a root folder
- scan recursively for `.zarr` datasets
- select all or clear all
- open only the checked datasets

The chosen `Visible sum`, `Truecolor`, `Raw spectral`, and preview options are applied to all selected Zarr datasets in the batch open action.

### Reader popup note

If a user opens `.zarr` files through napari's generic file-open dialog, napari may still show a `Choose reader` popup when multiple readers claim `.zarr`.

This plugin cannot reliably suppress that global napari chooser by itself.

The intended workaround is:

- use the plugin's own Zarr loader in `ND2 Spectral Export`
- open Zarr datasets from there instead of through napari's general file-open menu

### Visible sum definition

`Visible sum` is currently computed as the raw per-pixel mean intensity across spectral bins:

- sum all channel intensities at a pixel
- divide by the number of spectral bins
- do not normalize

So if a pixel has 24 spectral bins, the displayed gray value is:

`(bin_1 + bin_2 + ... + bin_24) / 24`

## Spectral Viewer Workflow

`Spectral Viewer` is where ROIs are drawn and spectral datasets are captured.

### Important ROI logic

ROI layers are now handled per image, not globally.

- Each active spectral image gets its own Shapes layer named like `image_name ROI`
- ROI labels are displayed through a separate companion label layer
- ROI numbering resets per image, so image 1 can have `ROI 1..N` and image 2 can also have `ROI 1..N`
- ROIs from image 1 are not reused automatically for image 2

### Recommended step-by-step use

1. Select a spectral image layer.
2. Click `Prepare ROI Layer`.
3. Draw ROIs for that image only.
4. Click `Plot ROI Spectrum`.
5. The plugin stores that image's ROI spectra in memory as a dataset.
6. Move to the next image.
7. Click `Prepare ROI Layer` again for that image.
8. Draw a fresh set of ROIs starting from `ROI 1`.

If you want to redraw for the current image, click `Clear Active ROI`. That clears only the active image's ROI shapes and restarts numbering from `ROI 1`.

### Stored ROI datasets

When `Plot ROI Spectrum` is used, the selected ROI spectra are stored in memory for the current napari session.

- Stored datasets remain available even if the image layer is later closed
- Stored datasets can be exported from `Spectral Viewer`
- Stored datasets are consumed by the `Spectral Analysis` panel

## Spectral Analysis Workflow

`Spectral Analysis` is intended for multi-image and multi-animal experiments.

### Metadata editing

The `Stored ROI Datasets` table lets you annotate each captured dataset with:

- `animal_id`
- `group_label`
- `genotype`
- `sex`
- `age`
- `region`
- `batch`
- `blind_id`

This supports experiments such as:

- 10 images total
- 5 WT and 5 HNPP animals
- multiple myelin ROIs per image
- aggregation from ROI level to image level to animal level

### Dataset selection and removal

Analysis no longer uses every stored dataset automatically.

- Use the `use_for_analysis` checkbox column to choose which dataset IDs are included
- Click `Compute Spectral Analysis` to analyze only the checked datasets
- Click `Remove Selected Datasets` to delete all checked datasets from memory
- Click `Remove Current Row` to delete the currently selected dataset

Unchecked datasets stay in memory but are ignored by the analysis.

### Available analysis outputs

The panel computes:

- ROI-level ratio table
- image-level summary table
- animal-level summary table

Each table can be exported to CSV.

### Ratio and statistics

The analysis panel supports:

- user-defined split wavelength
- ratio modes such as above/below split intensity ratio
- optional normalization before ratio calculation
- WT vs HNPP comparison
- one-way ANOVA by selected factor
- blind PCA and clustering for unlabeled datasets

### Recommended experiment flow

1. Open spectral images in napari.
2. For image 1, prepare the ROI layer and draw multiple myelin ROIs.
3. Plot the ROI spectrum to store that image's ROI dataset.
4. Repeat for image 2, image 3, and so on.
5. Open `Spectral Analysis`.
6. Enter metadata for each stored dataset.
7. Check only the dataset IDs you want to compare.
8. Set the wavelength split point.
9. Compute the analysis.
10. Export ROI, image, or animal summary CSV files as needed.
