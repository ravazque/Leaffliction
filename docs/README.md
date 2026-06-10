*This project has been created as part of the 42 curriculum by ravazque, acerezo- and maanguit.*

---

## Description

Leaffliction is a **computer vision** project: a complete image-classification pipeline that detects plant diseases from photographs of leaves (apple and grape varieties), built in Python.

The pipeline covers four stages, each with its own program:

- **`Distribution.py`** analyzes the dataset, counting images per class and plotting pie/bar charts to expose class imbalance.
- **`Augmentation.py`** balances the dataset by generating variants of existing images (flip, rotation, blur, contrast, scaling, cropping, distortion).
- **`Transformation.py`** extracts visual features with PlantCV transformations (Gaussian blur, masking, ROI analysis, pseudolandmarks, color histograms).
- **`train.py` / `predict.py`** train a **convolutional neural network** on the augmented dataset and classify new leaf images, displaying the original, its transformation and the predicted disease.

The trained model and the augmented dataset are packaged and signed (`signature.txt` holds the archive's SHA-1) so the exact deliverables can be verified during evaluation.
