# Chest X-ray 17-Pathology Classifier

Desktop application for multi-label chest X-ray classification with a
PySide6 interface and TensorFlow/Keras training and inference pipeline.

> **Medical notice:** This project is research and decision-support software.
> It is not a medical device and must not be used as a diagnosis or as a
> substitute for a qualified clinician.

## Features

- Drag-and-drop or file-picker image upload.
- Background TensorFlow inference so the GUI remains responsive.
- Transfer-learning model using DenseNet121 or EfficientNetB0.
- Multi-label sigmoid output with confidence scores and configurable thresholds.
- Interactive horizontal confidence bars for every available pathology.
- Color-coded results: positive findings are highlighted in red and negative
  findings in green.
- Clinician confirmation/correction workflow stored in a local SQLite database.
- Background fine-tuning on pending clinician feedback.
- ZIP dataset extraction with checksum verification and archive path-traversal
  protection.
- Patient-aware train/validation/test splitting for metadata-based datasets.
- Portable model and feedback storage for Linux and Windows.
- Automated Ubuntu and Windows checks through GitHub Actions.

## Current dataset

The project directory contains an `archive.zip` folder-based dataset. It
contains 17 class folders:

```text
abscess
ards
atelectasis
atherosclerosis_of_the_aorta
cardiomegaly
emphysema
fracture
hydropneumothorax
hydrothorax
pneumonia
pneumosclerosis
post_inflammatory_changes
post_traumatic_ribs_deformation
sarcoidosis
scoliosis
tuberculosis
venous_congestion
```

The application reads `.jpg`, `.jpeg`, `.png`, and `.bmp` images from each
class folder. The folder names become the model labels and are saved in
`metadata.json` beside the trained checkpoint. This archive is separate from
the original Kaggle metadata-based dataset, so its labels must not be mixed
with the original 17-label schema.

The archive is large and should not be committed to GitHub. Store it locally
or use Git LFS/object storage.

## Project structure

```text
.
├── main.py                 # Desktop application entry point
├── api.py                  # KaggleHub access and verified URL downloads
├── dataset_manager.py      # ZIP verification, extraction, and folder loading
├── image_validation.py     # Upload and prediction sanity checks
├── data_adapter.py         # Adapter for metadata/DataFrame datasets
├── data_pipeline.py        # TensorFlow input pipeline and patient splitting
├── labels.py               # Canonical metadata-dataset label order
├── model.py                # Model construction, metadata, prediction formatting
├── training.py             # Training, evaluation, and feedback fine-tuning
├── feedback.py             # SQLite clinician-feedback store
├── paths.py                # Linux/Windows application-data locations
├── ui/
│   ├── main_window.py      # PySide6 GUI
│   └── workers.py          # Background prediction/training workers
├── tests/                  # Unit and integration tests
├── requirements.txt
├── scripts/
│   └── prepare_data.py     # One-time ZIP extraction and label manifest
├── run_linux.sh            # Linux launcher with no-sudo Qt workaround
└── .github/workflows/ci.yml
```

## Requirements

- Python 3.11 or newer.
- PySide6.
- TensorFlow/Keras.
- NumPy, pandas, Pillow, scikit-learn, and pytest.
- At least several GB of free disk space for the dataset, extracted images,
  pretrained weights, and checkpoints.

TensorFlow installation support varies by operating system and Python
version. If the TensorFlow wheel is unavailable for your platform, use a
supported Python version or follow the TensorFlow installation instructions
for that platform.

## Linux installation

From the project directory:

```bash
sudo apt update
sudo apt install -y python3-venv libxcb-cursor0

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If you cannot use `sudo`, the included launcher downloads `libxcb-cursor0`
locally and adds it to `LD_LIBRARY_PATH`:

```bash
chmod +x run_linux.sh
./run_linux.sh
```

The local workaround is Debian/Ubuntu and x86_64 specific. The normal
`python3 main.py` launch is preferred when the system Qt libraries are
installed.

## Windows installation

Open PowerShell in the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again. The Linux-only `run_linux.sh` script is
not used on Windows.

## Launch the application

With the virtual environment activated:

```bash
python3 main.py
```

On Windows, use `python main.py`.

The application starts with **Predict** disabled until a trained checkpoint
exists. Prepare the archive into `data/` before training; the app reads the
prepared directory and never trains directly from the ZIP. Use **Choose Data
Folder** if your prepared directory is elsewhere.

## Train using the included ZIP

1. Start the application.
2. Prepare the archive once if `data/` does not exist:

   ```bash
   python3 -m scripts.prepare_data --archive archive.zip --output data
   ```

3. Start the app and confirm the dataset status shows `data/`.
4. Click **Train Model** and confirm the prompt.
5. Wait for preprocessing and training to finish.
6. Select or drop an X-ray image.
7. Click **Predict**.

Training downloads pretrained ImageNet weights when the selected backbone is
created for the first time. The first run may therefore require network access
and substantial disk space. Training runs in a worker thread, but the
operation can still take a long time.

The prepared-data workflow creates:

```text
data/
├── files/
├── labels.json
└── ...

<user-data-directory>/models/chest_xray/
├── best.keras
├── final.keras
├── metadata.json
└── dataset_manifest.json
```

## Prediction and clinician feedback

After prediction, the UI shows every model label with:

- confidence percentage,
- threshold,
- positive/negative status,
- horizontal confidence bar.

Uploads are checked before inference. Unsupported, corrupted, colour, tiny,
or unusually shaped files are rejected. A prediction with an indecisive model
output is rejected with:

> This image doesn't look like a valid chest X-ray for the 17 supported
> pathologies. Please upload a frontal chest X-ray.

For a valid scan where every probability is below the configurable detection
threshold, the app keeps the charts visible and shows a calm information box:

> No signs of any of the 17 supported pathologies, including tumor or
> cancer-related findings, were detected by the model.

It also reminds the user that this is automated output and that a doctor must
review the scan. Thresholds are defined in `image_validation.py`.

To contribute a correction:

1. Review the confidence bars.
2. Tick **Confirmed** for the labels that are clinically confirmed.
3. Click **Save Clinician Correction**.
4. Click **Fine-tune Feedback** when enough corrections are available.

Feedback images and `labels.csv` are stored under the local `feedback/`
directory, with SQLite metadata alongside them. Fine-tuning runs in a background worker
and marks used samples after a successful update. No feedback is uploaded by
this application.

Only use feedback from appropriately authorized and qualified reviewers. The
feedback loop does not replace proper validation, calibration, or governance.

## Model and dataset storage

The application uses `pathlib.Path` and platform-specific user-data folders:

| Platform | Default location |
| --- | --- |
| Linux | `$XDG_DATA_HOME/ChestXRayClassifier`, or `~/.local/share/ChestXRayClassifier` for models |
| Windows | `%APPDATA%\ChestXRayClassifier` |

Model files are stored under `models/chest_xray`. Feedback is stored in
Feedback is kept in the repository-local `feedback/` directory and is ignored
by Git. Set `XDG_DATA_HOME` on Linux if a different model location is required.

## Dataset integrity and API usage

`scripts/prepare_data.py` is the normal dataset entry point. It extracts the
archive once into `data/`, writes `labels.json`, and records the archive
checksum. Training reads the prepared directory and never extracts the ZIP.

`dataset_manager.py` provides:

- SHA-256 calculation.
- Expected-checksum verification.
- Safe ZIP extraction that rejects path traversal.
- Folder-based image/label discovery.
- Dataset manifest generation.

`api.py` provides:

- lazy KaggleHub loading through `load_dataset()`, with no network request on
  import;
- `download_dataset()` for URL-based archive download, checksum validation,
  and optional extraction.

Example:

```python
from pathlib import Path

from api import download_dataset

download_dataset(
    url="https://example.invalid/dataset.zip",
    destination=Path("data") / "dataset.zip",
    expected_sha256="<publish-the-real-sha256-here>",
    extract_to=Path("data") / "extracted",
)
```

Do not use a placeholder checksum in production. Publish the expected digest
through a trusted release or configuration channel.

## Metadata-based dataset training

For a DataFrame dataset exposed through `api.py`, use the adapter:

```python
from data_adapter import load_records
from training import train

frame, columns = load_records()
history = train(
    frame,
    image_column=columns.image,
    patient_column=columns.patient,
)
```

The metadata adapter expects an image-path column, a patient identifier
column, and the canonical pathology columns defined in `labels.py`.
Metadata-based splits are deterministic and patient-disjoint. The folder ZIP
workflow uses its folder names as labels and stores one image record per
folder/image pair.

## Testing

Run the test suite from the activated environment:

```bash
python3 -m pytest -q
```

Compile the application source:

```bash
find . \
  -path './.venv' -prune -o \
  -path './.qt-libs' -prune -o \
  -name '*.py' -print0 | xargs -0 python3 -m py_compile
```

For headless Linux machines or CI:

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
```

The test suite covers label validation, patient splitting, prediction
formatting, archive/folder loading, checksums, and SQLite feedback storage.
It intentionally does not download the full dataset or run a full training
job.

## Continuous integration

GitHub Actions is configured in `.github/workflows/ci.yml` and checks:

- Ubuntu and Windows;
- Python compilation;
- unit tests;
- PySide6 GUI construction.

The CI job does not download the dataset, pretrained weights, or the full
TensorFlow runtime. Add a separate self-hosted or scheduled training workflow
if reproducible model training is required.

## Packaging

The code uses `pathlib`, lazy imports for heavy dependencies, and user-data
directories suitable for PyInstaller. A typical build flow is:

```bash
python -m pip install pyinstaller
pyinstaller --noconfirm --windowed --name ChestXRayClassifier main.py
```

Before distributing a packaged build, verify that:

- TensorFlow shared libraries are collected;
- the PySide6 platform plugins are included;
- model/data directories are writable;
- the first-run dataset and pretrained-weight behavior is documented;
- Windows and Linux builds are produced separately.

## Limitations and production checklist

- A trained checkpoint is required before prediction.
- The included ZIP labels are not interchangeable with the canonical metadata
  labels from the Kaggle adapter.
- Full training is compute- and storage-intensive.
- Confidence values are model probabilities, not guaranteed clinical
  likelihoods.
- Thresholds are currently metadata-driven and require calibration for a
  clinical deployment.
- The application has no authentication, audit server, PACS integration, or
  remote feedback service.
- Validate patient identifiers and split policy against the source dataset
  before using results for research.
- Add independent clinical validation, calibration, monitoring, privacy
  controls, and regulatory review before any real-world deployment.

## License and data

Add the project license and dataset license here before publishing the
repository. Do not redistribute `archive.zip` unless its dataset terms permit
redistribution. Ensure that any X-ray images and clinician feedback comply
with applicable privacy, consent, and data-protection requirements.
