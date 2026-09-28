import hashlib
import os
import shutil
import subprocess
import zipfile

import numpy as np
import pytest
import torch

from leaffliction.cli import LeafError
from leaffliction.learnings import (MARKER, build_zip, load_learnings,
                                    prepare_workdir, save_model,
                                    write_signature)
from leaffliction.model import MIN_SIZE, LeafCNN
from leaffliction.training import pick_device, predict, report

CLASSES = ["a", "b", "c"]


@pytest.fixture
def bundle(tmp_path):
    """A saved work directory, its zip and signature."""
    work = str(tmp_path / "learnings")
    prepare_workdir(work)
    os.makedirs(os.path.join(work, "train", "a"))
    with open(os.path.join(work, "train", "a", "x.JPG"), "wb") as f:
        f.write(b"fake")
    torch.manual_seed(0)
    model = LeafCNN(len(CLASSES)).eval()
    save_model(work, model, {"classes": CLASSES, "img_size": MIN_SIZE})
    zip_path = str(tmp_path / "learnings.zip")
    build_zip(work, zip_path)
    return model, work, zip_path


@pytest.mark.parametrize("size", [MIN_SIZE, 64, 97])
def test_model_outputs_one_score_per_class(size):
    out = LeafCNN(5).eval()(torch.zeros(2, 3, size, size))
    assert out.shape == (2, 5)


def test_zip_layout_and_marker_left_out(bundle):
    _, _, zip_path = bundle
    names = zipfile.ZipFile(zip_path).namelist()
    assert "learnings/meta.json" in names
    assert "learnings/model.pt" in names
    assert "learnings/train/a/x.JPG" in names
    assert not any(n.endswith(MARKER) for n in names)


@pytest.mark.parametrize("source", ["zip", "dir"])
def test_learnings_round_trip(bundle, source):
    model, work, zip_path = bundle
    loaded, meta = load_learnings(zip_path if source == "zip" else work)
    assert meta["classes"] == CLASSES
    x = torch.randint(0, 256, (4, 3, MIN_SIZE, MIN_SIZE), dtype=torch.uint8)
    cpu = torch.device("cpu")
    assert np.allclose(predict(model, x, cpu), predict(loaded, x, cpu))


def test_signature_matches_sha1_and_sha1sum(bundle, tmp_path):
    _, _, zip_path = bundle
    sig = str(tmp_path / "signature.txt")
    digest = write_signature(zip_path, sig)
    with open(zip_path, "rb") as f:
        assert digest == hashlib.sha1(f.read()).hexdigest()
    with open(sig) as f:
        assert f.read() == f"{digest}  learnings.zip\n"
    if shutil.which("sha1sum"):
        subprocess.run(["sha1sum", "-c", sig], cwd=tmp_path, check=True,
                       capture_output=True)


def test_load_learnings_errors(tmp_path, bundle):
    with pytest.raises(LeafError, match="run train.py"):
        load_learnings(str(tmp_path / "missing.zip"))
    text = tmp_path / "notes.txt"
    text.write_text("x")
    with pytest.raises(LeafError, match="neither"):
        load_learnings(str(text))
    empty = str(tmp_path / "empty.zip")
    zipfile.ZipFile(empty, "w").close()
    with pytest.raises(LeafError, match="invalid learnings"):
        load_learnings(empty)
    _, work, _ = bundle
    with open(os.path.join(work, "model.pt"), "wb") as f:
        f.write(b"garbage")
    with pytest.raises(LeafError, match="invalid learnings"):
        load_learnings(work)


def test_prepare_workdir_only_wipes_its_own_folders(tmp_path):
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("precious")
    with pytest.raises(LeafError, match="not created by train.py"):
        prepare_workdir(str(foreign))
    assert (foreign / "keep.txt").exists()
    own = str(tmp_path / "own")
    prepare_workdir(own)
    open(os.path.join(own, "old.txt"), "w").close()
    prepare_workdir(own)
    assert os.listdir(own) == [MARKER]


def test_report_counts_and_accuracy():
    accuracy, text = report([0, 1, 1, 2], [0, 1, 2, 2], CLASSES)
    assert accuracy == 0.75
    assert "Accuracy: 75.00% (3/4 images)" in text
    assert "50.00%" in text


def test_pick_device_validation():
    assert pick_device("cpu").type == "cpu"
    with pytest.raises(LeafError):
        pick_device("not-a-device")
