# Leaffliction

Leaffliction is a computer vision project: an image-classification pipeline
that recognises plant diseases from photographs of leaves, written in Python.
The dataset holds 7 221 photos of apple and grape leaves in eight classes,
from 275 `Apple_rust` to 1 640 `Apple_healthy`, and a convolutional neural
network trained on it names the disease of a leaf it has never seen.

## How it works

| Program | Role |
|---|---|
| `Distribution.py` | Counts the images of every class and draws, per plant type, a pie chart and a bar chart |
| `Augmentation.py` | Six augmentations of one image (flip, rotate, skew, shear, crop, distortion), or a copy of the dataset balanced with them |
| `Transformation.py` | PlantCV views of a leaf: Gaussian blur, mask, region of interest, shape analysis, pseudolandmarks, colour histograms |
| `train.py` | Splits, balances, trains the network and packs the result into `learnings.zip`, signed in `signature.txt` |
| `predict.py` | Classifies one image and shows it with its transformed leaf, or scores a whole folder |

- **Segmentation.** The leaf is cut out on the LAB `b - a` channel, where
  leaves (green to brown) and backgrounds (grey or purple) separate: an Otsu
  threshold, hysteresis to keep lesions and glare, morphology, the largest
  component and hole filling. The network only sees the leaf on white, so it
  cannot learn the photo setup instead of the disease.
- **Split before augmenting.** A photo, its augmented copies and its
  byte-identical duplicates form one group, and a whole group lands on one side
  of a stratified 80 / 20 split. Validation keeps original images only and is
  never used to tune anything, so its score is free of data leakage.
- **Balancing.** Each training class is filled with augmentations of its own
  originals, every original getting each technique at most once; uncovered
  areas take the background colour so the mask ignores them.
- **Network.** `LeafCNN`: four blocks of two 3x3 convolutions with batch
  normalisation and ReLU, each followed by max pooling, then global average
  pooling, dropout and one linear layer, 1.2 M parameters, on 128x128 inputs.
  It is trained for 20 epochs with AdamW, a one-cycle learning rate up to
  0.002, label smoothing, random flips and quarter turns.
- **Learnings.** `learnings.zip` holds the weights, the metadata, the report
  and the images trained on; `predict.py` reads the model straight from the
  zip, and `signature.txt` holds its SHA-1. Neither the dataset nor the archive
  is in the repository: the archive is a release asset of the project.

On the 1 445 validation images the packaged model scores 100.00 %, every class
at 100 %, after a training of about 3 minutes on a laptop GPU.

## Dependencies

| Dependency | Version | Debian / Ubuntu | Arch |
|---|---|---|---|
| Python 3.13, or `uv` | 3.13 exactly: PlantCV pins `scipy<1.16`, which has no 3.14 wheels | `python3.13 python3.13-venv` | `uv`, which downloads 3.13 itself |
| GNU Make, `curl`, `unzip` | any | `make curl unzip` | `make curl unzip` |
| PyTorch, PlantCV, OpenCV, NumPy, Matplotlib | pinned in `requirements.txt` | installed into `.venv/` by `make install` | same |
| Tk | optional, for the figure windows | `python3.13-tk` | included in uv's Python |
| A CUDA GPU | optional: everything runs on the CPU, only training is slow there | | |

The CPU build of PyTorch takes about 1.5 GB and is the default; `TORCH=cuda`
selects the CUDA build, about 6 GB.

## Build and run

```bash
make install                                                          # .venv with Python 3.13 and requirements.txt
make learnings                                                        # downloads learnings.zip and checks it against signature.txt
make predict                                                          # classifies one leaf, then scores the 1445 validation images
make predict IMG="learnings/validation/Grape_spot/image (1000).JPG"
make test                                                             # 107 pytest tests on synthetic leaves, no dataset needed
make lint                                                             # flake8 (PEP 8)
make clean                                                            # generated figures and folders
make fclean                                                           # clean, the .venv and the extracted learnings/; learnings.zip is kept
```

With the dataset extracted in `images/` (one folder per class):

```bash
make distribution     # class counts and charts
make augmentation     # balanced copy of the dataset in augmented_directory/
make transformation   # PlantCV views of the first image
make train            # trains and writes a new learnings.zip and signature.txt
```

`make` alone lists the targets. Every program also runs on its own with the
environment active, prints its options with `-h`, and saves its figures as PNG
when there is no display. A new training is a different archive with a
different SHA-1. Errors exit with `1`, invalid arguments with `2` and `Ctrl+C`
with `130`, always with a message and never a traceback.

## Layout

```
Leaffliction/
├── Makefile           # install, learnings, predict, the programs, test, lint
├── requirements.txt   # pinned dependencies
├── signature.txt      # SHA-1 of learnings.zip
├── src/               # the five programs and the leaffliction/ package they share
├── tests/             # pytest suite on synthetic leaves
└── docs/              # documentation and the figures drawn by the programs
```
