"""End-to-end runs of the five programs, headless, including bad input.

The cli fixture fails any run that prints a Python traceback.
"""
import os
import subprocess

import numpy as np
import pytest
from conftest import leaf_image, write

from leaffliction.augment import AUGMENTATIONS

BAD_PATHS = ["missing", "notes.txt"]


def bad_path(tmp_path, kind):
    if kind == "notes.txt":
        (tmp_path / kind).write_text("not an image")
    return kind


# Distribution

def test_distribution_charts_each_plant(cli, dataset, tmp_path):
    result = cli("Distribution.py", dataset)
    assert result.returncode == 0
    assert "Plant: 2 classes, 20 images" in result.stdout
    assert (tmp_path / "Plant_distribution.png").exists()


@pytest.mark.parametrize("kind", BAD_PATHS + ["empty"])
def test_distribution_rejects_bad_directories(cli, tmp_path, kind):
    if kind == "empty":
        (tmp_path / kind).mkdir()
    else:
        bad_path(tmp_path, kind)
    result = cli("Distribution.py", kind)
    assert result.returncode == 1 and result.stderr.startswith("Error:")


def test_distribution_without_argument_prints_usage(cli):
    result = cli("Distribution.py")
    assert result.returncode == 2 and "usage:" in result.stderr


# Augmentation

def test_augmentation_saves_six_images_next_to_original(cli, leaf_file):
    result = cli("Augmentation.py", leaf_file)
    assert result.returncode == 0
    folder = os.path.dirname(leaf_file)
    expected = {f"leaf (1)_{n}.JPG" for n in AUGMENTATIONS}
    assert expected <= set(os.listdir(folder))


def test_augmentation_balances_a_directory(cli, dataset, tmp_path):
    os.remove(os.path.join(dataset, "Plant_spot", "image (0).JPG"))
    result = cli("Augmentation.py", dataset, "--dst", "balanced")
    assert result.returncode == 0
    for label in ("Plant_healthy", "Plant_spot"):
        assert len(os.listdir(tmp_path / "balanced" / label)) == 10
    assert len(os.listdir(os.path.join(dataset, "Plant_spot"))) == 9


def test_augmentation_refuses_unsafe_destinations(cli, dataset, tmp_path):
    inside = os.path.join(dataset, "out")
    assert cli("Augmentation.py", dataset, "--dst", inside).returncode == 1
    assert cli("Augmentation.py", os.path.join(dataset, "Plant_spot"),
               "--dst", dataset).returncode == 1
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    result = cli("Augmentation.py", dataset, "--dst", "busy")
    assert result.returncode == 1 and "already exists" in result.stderr


@pytest.mark.parametrize("kind", BAD_PATHS)
def test_augmentation_rejects_bad_images(cli, tmp_path, kind):
    result = cli("Augmentation.py", bad_path(tmp_path, kind))
    assert result.returncode == 1 and result.stderr.startswith("Error:")


def test_augmentation_rejects_bad_target(cli, dataset):
    assert cli("Augmentation.py", dataset, "--target", "0").returncode == 2


# Transformation

def test_transformation_displays_one_image(cli, leaf_file, tmp_path):
    result = cli("Transformation.py", leaf_file)
    assert result.returncode == 0
    assert (tmp_path / "transformations.png").exists()
    assert (tmp_path / "histogram.png").exists()


def test_transformation_saves_selected_flags(cli, dataset, tmp_path):
    write(os.path.join(dataset, "Plant_spot", "blank.png"),
          np.zeros((40, 40, 3), np.uint8))
    result = cli("Transformation.py", "-src", dataset, "-dst", "out",
                 "-mask", "-hist")
    assert result.returncode == 0
    assert "no leaf found" in result.stderr
    files = os.listdir(tmp_path / "out" / "Plant_spot")
    assert "image (3)_Mask.JPG" in files
    assert "image (3)_Histogram.png" in files
    assert not any("Blur" in f for f in files)


@pytest.mark.parametrize("args, code", [
    ([], 2),
    (["-src", "x"], 2),
    (["img.JPG", "-src", "x", "-dst", "y"], 2),
    (["-src", "missing", "-dst", "out"], 1),
    (["missing.JPG"], 1),
    (["-h"], 0),
])
def test_transformation_argument_errors(cli, args, code):
    assert cli("Transformation.py", *args).returncode == code


def test_transformation_refuses_dst_inside_src(cli, dataset):
    result = cli("Transformation.py", "-src", dataset, "-dst",
                 os.path.join(dataset, "out"))
    assert result.returncode == 1


# train and predict

def test_train_then_predict(cli, dataset, tmp_path):
    result = cli("train.py", dataset, "--epochs", "2", "--img-size", "32",
                 "--device", "cpu", "--val-split", "0.3")
    assert result.returncode == 0, result.stderr
    assert "Accuracy:" in result.stdout
    assert "fewer than 100" in result.stderr
    subprocess.run(["sha1sum", "-c", "signature.txt"], cwd=tmp_path,
                   check=True, capture_output=True)
    val = tmp_path / "learnings" / "validation"
    assert sorted(os.listdir(val)) == ["Plant_healthy", "Plant_spot"]
    assert len(os.listdir(tmp_path / "learnings" / "train" / "Plant_spot")) \
        == len(os.listdir(tmp_path / "learnings" / "train" / "Plant_healthy"))

    image = next(iter(val.glob("Plant_spot/*.JPG")))
    single = cli("predict.py", image, "--device", "cpu")
    assert single.returncode == 0
    assert single.stdout.startswith("Class predicted: Plant_")
    assert (tmp_path / "prediction.png").exists()

    batch = cli("predict.py", val, "--device", "cpu")
    assert batch.returncode == 0 and "Accuracy:" in batch.stdout
    unknown = tmp_path / "unknown"
    write(str(unknown / "leaf.JPG"), leaf_image()[0])
    listed = cli("predict.py", unknown, "--learnings", "learnings")
    assert listed.returncode == 0 and "leaf.JPG: Plant_" in listed.stdout


@pytest.mark.parametrize("args", [
    ["--val-split", "1"], ["--img-size", "8"], ["--epochs", "0"]])
def test_train_argument_errors(cli, dataset, args):
    assert cli("train.py", dataset, *args).returncode == 2


def test_train_needs_two_classes_and_a_safe_workdir(cli, dataset, tmp_path):
    single = cli("train.py", os.path.join(dataset, "Plant_spot"))
    assert single.returncode == 1 and "2 classes" in single.stderr
    (tmp_path / "learnings").mkdir()
    (tmp_path / "learnings" / "precious.txt").write_text("x")
    busy = cli("train.py", dataset, "--device", "cpu")
    assert busy.returncode == 1 and "not created by" in busy.stderr
    inside = cli("train.py", dataset, "--workdir",
                 os.path.join(dataset, "work"))
    assert inside.returncode == 1


def test_predict_errors(cli, tmp_path, leaf_file):
    missing = cli("predict.py", leaf_file)
    assert missing.returncode == 1 and "run train.py" in missing.stderr
    (tmp_path / "fake.zip").write_text("x")
    fake = cli("predict.py", leaf_file, "--learnings", "fake.zip")
    assert fake.returncode == 1
    assert cli("predict.py", "missing.JPG").returncode == 1
