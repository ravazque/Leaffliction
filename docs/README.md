# Leaffliction

## Description

Leaffliction is a **computer vision** project: a complete image-classification pipeline that detects plant diseases from photographs of leaves (apple and grape varieties), built in Python.

The pipeline covers four stages, each with its own program:

- **`Distribution.py`** analyzes the dataset, counting images per class and plotting pie/bar charts to expose class imbalance.
- **`Augmentation.py`** balances the dataset by generating geometric variants of existing images (flip, rotate, skew, shear, crop, distortion).
- **`Transformation.py`** extracts visual features with PlantCV (Gaussian blur, mask, ROI objects, object analysis, pseudolandmarks, colour histograms).
- **`train.py` / `predict.py`** train a **convolutional neural network** on background-free leaves and classify new images, showing the original, its transformation and the predicted disease.

The trained model and the images it learned from are packaged in `learnings.zip`, and `signature.txt` holds that archive's SHA-1 so the exact artifacts can be verified later.

---

## Contents

1. [Setup](#setup)
2. [Dataset layout](#dataset-layout)
3. [Usage](#usage)
4. [Concepts](#concepts)
5. [Results](#results)
6. [Testing](#testing)
7. [Project structure](#project-structure)

---

## Setup

Requirements: **Python 3.13** (the tested version). Python 3.14 is not supported yet: PlantCV pins `scipy<1.16`, which ships no 3.14 wheels. A CUDA GPU is optional; training falls back to the CPU.

```bash
python3.13 -m venv .venv            # or: uv venv --python 3.13 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every command below assumes the virtual environment is active and is run from the repository root.

Figures open in a window. Without a display (SSH, CI, `MPLBACKEND=Agg`), each figure is saved as a PNG in the current directory instead and its path is printed.

---

## Dataset layout

Every directory that directly holds images is a **class**, named after the directory. Two layouts are understood:

```
images/                      Apple/
├── Apple_Black_rot/         ├── apple_healthy/
├── Apple_healthy/           ├── apple_scab/
├── ...                      └── ...
└── Grape_spot/
```

The **plant type** of a class is its parent folder when nested (`Apple/apple_scab` → `Apple`), otherwise the part of its name before the first underscore (`Apple_scab` → `Apple`). Supported extensions: `.jpg .jpeg .png .bmp .tif .tiff`, in any letter case. Hidden files and folders are ignored.

The dataset used here has 7221 RGB images of 256×256 pixels in 8 classes, heavily imbalanced (from 275 `Apple_rust` to 1640 `Apple_healthy`).

---

## Usage

Every program prints its options with `-h`.

### Distribution

```bash
./src/Distribution.py images/
./src/Distribution.py images/Apple_rust         # a single class also works
```

Prints a count table and shows, per plant type, a pie chart (share of each class) and a bar chart (image count). The same class keeps the same colour in both charts. Plants with more than 8 classes fold the smallest ones into "Other".

### Augmentation

```bash
./src/Augmentation.py "images/Apple_healthy/image (1).JPG"
./src/Augmentation.py images/                   # writes augmented_directory/
```

| Argument | Meaning |
|---|---|
| `path` | an image, or a dataset directory |
| `--dst DIR` | output folder (image: next to the original; directory: `augmented_directory`) |
| `--target N` | images per class when balancing (default: the largest class) |
| `--seed N` | random seed (default 7) |

- **Image**: saves the 6 augmentations as `<name>_Flip.JPG`, `<name>_Rotate.JPG`, … next to the original and displays them.
- **Directory**: copies the dataset into `--dst` and adds augmented images until every class reaches the target. The destination must be empty and must not overlap the source.

### Transformation

```bash
./src/Transformation.py "images/Apple_healthy/image (1).JPG"
./src/Transformation.py -src images/Apple_healthy -dst transformed
./src/Transformation.py -src images/Apple_healthy -dst transformed -mask -hist
```

| Argument | Meaning |
|---|---|
| `image` | display the transformations of one image |
| `-src PATH` | image or directory to process (recursive, sub-folders mirrored) |
| `-dst DIR` | where to save; required with `-src` |
| `-blur -mask -roi -analyze -pseudo -hist` | select transformations (default: all) |

Saved files are named `<name>_<Transformation>.JPG`, plus `<name>_Histogram.png`. Images where no leaf is found are skipped with a warning.

### train

```bash
./src/train.py images/
```

| Option | Default | Meaning |
|---|---|---|
| `--out` | `learnings.zip` | archive to produce |
| `--signature` | `signature.txt` | SHA-1 file of the archive |
| `--workdir` | `learnings` | folder that gets zipped |
| `--epochs` | 20 | training epochs |
| `--batch-size` | 64 | images per optimisation step |
| `--img-size` | 128 | network input side (pixels) |
| `--lr` | 0.002 | peak learning rate |
| `--val-split` | 0.2 | share of leaves kept for validation |
| `--seed` | 7 | seed for the split, augmentation and weights |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` |

About 3–4 minutes on a laptop GPU (RTX 4050), mostly spent segmenting the images. The work directory is only deleted if a previous `train.py` run created it.

### predict

```bash
./src/predict.py "learnings/validation/Grape_spot/image (1000).JPG"
./src/predict.py learnings/validation            # scores a whole folder
```

| Option | Default | Meaning |
|---|---|---|
| `--learnings` | `learnings.zip` | archive (or extracted folder) holding the model |
| `--device` | `auto` | as in `train.py` |

- **Image**: prints `Class predicted: <class> (<confidence>)` and shows the original, the transformed leaf and the prediction.
- **Directory**: predicts every image. When folder names are known classes, it prints accuracy, per-class accuracy, the confusion matrix and the misclassified files; otherwise it lists one prediction per image.

### Exit codes

`0` success, `1` error (message on stderr, never a traceback), `2` invalid arguments, `130` interrupted with Ctrl-C.

---

## Concepts

### 1. Class imbalance

A classifier trained on imbalanced data learns the prior: predicting the majority class is "right" most of the time. Here `Apple_healthy` alone is 23 % of the images and `Apple_rust` under 4 %. `Distribution.py` makes this visible; `Augmentation.py` and `train.py` correct it.

### 2. Data augmentation

Augmentation creates new, plausible training images from existing ones. It **balances** classes and **regularises** the network: it cannot memorise one exact picture if it sees variants of it.

| Technique | What it does | Why it is plausible for a leaf |
|---|---|---|
| Flip | mirror horizontally, vertically or both | leaves have no fixed orientation |
| Rotate | rotate 15–45° either way | a photo can be taken at any angle |
| Skew | projective tilt: one edge shrinks toward its centre | camera not parallel to the leaf |
| Shear | slant along x or y | same, with an affine model |
| Crop | keep 70–85 % of the image, scale back up | zoom, partial framing |
| Distortion | smooth sinusoidal warp of the pixel grid | leaves are not flat |

Areas uncovered by a transform are filled with the image's median border colour (the background) rather than black or a mirror copy, so the segmentation never mistakes them for leaf.

Each result is reproducible: the random generator is seeded from the seed, the file and the technique.

**Balancing.** Each class is filled up to the target using only original images as sources. Sources are shuffled, and augmentations are assigned in rounds so that every source gets each technique at most once and the techniques stay evenly spread. A class can therefore grow to at most 7× its originals; a warning says so if the target is out of reach. Names already present are never overwritten.

### 3. Image transformation (feature extraction)

**Colour spaces.** Images are stored as RGB, but other spaces separate what matters better:

- **HSV**: hue (colour tone), saturation (purity), value (brightness).
- **LAB**: lightness, **a** (green ↔ magenta) and **b** (blue ↔ yellow), designed so that distances match perceived colour differences.

**Leaf segmentation.** Leaves are green, yellow or brown: high on *b*, low on *a*. The grey or purple backgrounds are the opposite. The pipeline works on the single channel `b − a`, where leaf and background form two clear populations:

1. **Otsu threshold**: picks the cut that best separates the two populations of the histogram (it maximises the variance between them). Pixels above it are *sure leaf*.
2. **Hysteresis**: pixels between the background level and the Otsu cut are kept only when connected to sure leaf. This recovers lesions and pale or glossy areas without letting the background in.
3. **Gaussian blur**: a Gaussian kernel averages each pixel with its neighbours, weighted by distance. Blurring the binary mask and thresholding it again smooths jagged edges. This is the *Gaussian blur* view.
4. **Morphology**: an *opening* (erode, then dilate) removes specks; a *closing* (dilate, then erode) seals thin gaps.
5. **Largest connected component**: keeps the leaf and drops isolated blobs.
6. **Hole filling**: dark lesions inside the leaf become part of it.

A channel with almost no contrast (a blank image) is reported as "no leaf". Known limitation: a strong specular highlight on the very edge of a leaf can be cut out, since it is as colourless as the background.

**The views.**

| View | Content |
|---|---|
| Gaussian blur | the smoothed binary mask (white = leaf) |
| Mask | the leaf on a white background; this is also the input of the network |
| ROI objects | objects intersecting a rectangular region of interest, tinted green; the ROI in blue |
| Analyze object | PlantCV shape analysis: outline, convex hull, width, height, centre of mass, longest axis |
| Pseudolandmarks | points spaced along the x axis on the top edge, bottom edge and centre line of the leaf, used to compare shapes |
| Colour histogram | for each channel of RGB, HSV and LAB, the share of leaf pixels at each intensity. Diseases shift these curves: brown lesions raise red and push *a* from green toward magenta |

### 4. Classification

**Preprocessing.** One function turns an image into a network input, for training and for prediction alike: segment → remove background → resize to 128×128 → scale to [-1, 1]. Using the same function in both places prevents a mismatch between what the model learned and what it is shown.

**Split before augmenting.** The split is made **per leaf** and is stratified per class (80 % training, 20 % validation):

- a file and its augmented copies (`x.JPG`, `x_Flip.JPG`, …) share one group, recognised by the suffixes;
- byte-identical photos (the dataset has 7 such pairs) also share one group;
- a whole group lands on one side.

Validation keeps original images only. Augmenting first and splitting afterwards would put copies of the same leaf on both sides. That is **data leakage**, and it inflates the validation score. Only the training part is balanced.

**Network (`LeafCNN`, 1.2 M parameters).**

| Stage | Layers | Output (128 px input) |
|---|---|---|
| Block 1 | 2 × (Conv 3×3 → BatchNorm → ReLU) → MaxPool 2 | 32 × 64 × 64 |
| Block 2 | same | 64 × 32 × 32 |
| Block 3 | same | 128 × 16 × 16 |
| Block 4 | same | 256 × 8 × 8 |
| Head | global average pool → Dropout 0.3 → Linear | one score per class |

- A **convolution** slides small learned filters over the image. Early filters respond to edges and colour; deeper ones to textures such as spots, rust pustules or rot.
- **Batch normalisation** re-centres activations batch by batch, which makes training faster and more stable.
- **ReLU** keeps positive values and zeroes the rest, making the network non-linear.
- **Max pooling** halves the resolution, keeping the strongest responses. This grows the area each filter sees.
- **Global average pooling** summarises each feature map into one number, independent of position.
- **Dropout** randomly silences features during training, so the network cannot rely on a single one.
- **Softmax** turns the final scores into probabilities; the highest is the prediction and its confidence.

**Training.**

- **Cross-entropy loss** with label smoothing 0.05: it penalises low probability on the true class, and the smoothing discourages over-confidence.
- **AdamW** optimiser (adaptive steps, decoupled weight decay).
- **One-cycle** learning-rate schedule: warm-up to 0.002, then decay.
- Random flips and quarter turns on each batch.
- Mixed precision on GPU.

The number of epochs is fixed in advance. The validation set is never used to choose a checkpoint or tune anything, so its score is an honest estimate of performance on unseen leaves.

**Overfitting** shows up as training accuracy far above validation accuracy. It is limited here by the per-leaf split, augmentation, dropout, weight decay, label smoothing and background removal. Background removal matters because it stops the network from learning shortcuts from the photo setup (background colour, shadows) instead of the disease.

**Metrics.**

- Accuracy: share of correct predictions.
- Per-class accuracy: exposes a weak class that a global number hides.
- Confusion matrix: rows = true class, columns = predicted, so errors between similar diseases stand out.

### 5. Learnings archive and signature

```
learnings.zip
└── learnings/
    ├── model.pt          network weights (PyTorch state dict)
    ├── meta.json         class names, input size, seed, split sizes, validation accuracy
    ├── metrics.txt       validation report
    ├── train/<class>/    balanced training set: augmented + background removed
    └── validation/<class>/  untouched validation images
```

`predict.py` reads the model straight from the zip, so the prediction always comes from the signed file. `signature.txt` uses the `sha1sum` format (`<sha1>  learnings.zip`):

```bash
sha1sum -c signature.txt          # Linux
shasum -c signature.txt           # macOS
```

Any change to the archive, even re-zipping the same content, changes the hash. Keep `learnings.zip` untouched after training. Neither the dataset nor the archive belongs in the repository; only `signature.txt` does.

---

## Results

`./src/train.py images/` with the default settings (seed 7, 20 epochs, 128 px), on an RTX 4050 laptop GPU, in 3 min 18 s:

| Split | Images | Accuracy |
|---|---|---|
| Training (balanced, augmented, measured during training with dropout and flips) | 10 488 | 99.98 % |
| **Validation (original images, leaves never seen in training)** | **1 445** | **100.00 %** |

| Class | Validation images | Accuracy |
|---|---|---|
| Apple_Black_rot | 124 | 100 % |
| Apple_healthy | 329 | 100 % |
| Apple_rust | 55 | 100 % |
| Apple_scab | 126 | 100 % |
| Grape_Black_rot | 236 | 100 % |
| Grape_Esca | 276 | 100 % |
| Grape_healthy | 84 | 100 % |
| Grape_spot | 215 | 100 % |

The confusion matrix is diagonal. A second run with a different seed, which gives a different split and different initial weights, also scored 100 % (1443/1443). The photos are taken in controlled conditions, one leaf on a plain background, and each disease has a distinctive texture; with the background removed, the classes separate cleanly. Accuracy on field photos (cluttered backgrounds, several leaves, other lighting) would be lower, since the segmentation assumes one leaf on a uniform background.

To reproduce the check on the packaged model:

```bash
sha1sum -c signature.txt
./src/predict.py learnings/validation
```

---

## Testing

### Automated

```bash
python -m pytest -q          # full suite, about 1 minute
python -m pytest tests/test_cli.py -q
flake8 src tests             # style (PEP 8)
```

The suite builds synthetic leaves, so it does not depend on the dataset. The only exception is one check on real images, which is skipped if `images/` is absent.

| File | Covers |
|---|---|
| `test_dataset.py` | extensions and hidden files, flat and nested layouts, duplicate folder names, augmented-name parsing, per-leaf split without leakage, duplicate photos, invalid roots |
| `test_augment.py` | every technique on square, rectangular and tiny images; reproducibility; balancing plan (target, cap, existing names, originals only); copy/augment job errors |
| `test_imageio.py` | grayscale and RGBA input, spaces and non-ASCII paths, empty/corrupt/truncated files, directories, unknown extensions |
| `test_mask_transform.py` | mask accuracy on synthetic leaves (IoU), plausible masks on real images, uniform images, view sizes, histogram percentages, saved file names, skipped files |
| `test_learnings.py` | model output shape, zip layout, zip/folder round trip, SHA-1 and `sha1sum -c`, corrupt or missing archives, work-directory safety, report, device parsing |
| `test_cli.py` | the five programs end to end, headless: normal runs, missing/invalid paths, empty folders, unsafe destinations, argument errors, and a full train → predict run on CPU. Any traceback fails the test |

### Manual checks

```bash
./src/Distribution.py images/                          # 2 figures, imbalance visible
./src/Augmentation.py images/                          # then:
./src/Distribution.py augmented_directory/             # 8 × 1640, 25 % each
cp "images/Apple_scab/image (12).JPG" /tmp/leaf.JPG
./src/Augmentation.py /tmp/leaf.JPG && ls /tmp/leaf_*.JPG
./src/Transformation.py "images/Grape_Esca/image (5).JPG"
./src/Transformation.py -src images/Apple_rust -dst /tmp/rust -mask -pseudo
./src/train.py images/
./src/predict.py learnings/validation                  # accuracy on ≥ 100 images
./src/predict.py "learnings/validation/Grape_spot/$(ls learnings/validation/Grape_spot | head -1)"
sha1sum -c signature.txt
```

---

## Project structure

```
.
├── requirements.txt
├── signature.txt               SHA-1 of learnings.zip
├── src/
│   ├── Distribution.py         class counts and charts
│   ├── Augmentation.py         single-image augmentation / dataset balancing
│   ├── Transformation.py       PlantCV views, displayed or saved
│   ├── train.py                split, balance, train, package, sign
│   ├── predict.py              classify one image or score a folder
│   └── leaffliction/
│       ├── cli.py              error handling, exit codes, figure display
│       ├── imageio.py          reading/writing images (BGR, any path)
│       ├── dataset.py          classes, plant types, leaf groups, split
│       ├── augment.py          the six augmentations
│       ├── balance.py          balancing plan and file jobs
│       ├── parallel.py         process pool with progress
│       ├── mask.py             leaf segmentation
│       ├── transform.py        the PlantCV views
│       ├── histogram.py        colour histograms
│       ├── figures.py          all matplotlib figures
│       ├── data.py             network input preparation
│       ├── model.py            LeafCNN
│       ├── training.py         training loop, inference, report
│       └── learnings.py        work directory, zip, signature, loading
└── tests/                      pytest suite
```
