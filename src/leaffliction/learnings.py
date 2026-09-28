"""The learnings bundle: model, metadata and images, zipped and signed."""
import hashlib
import io
import json
import os
import pickle
import shutil
import zipfile

import torch

from .cli import LeafError
from .dataset import is_image
from .model import LeafCNN

MODEL, META = "model.pt", "meta.json"
MARKER = ".learnings"  # marks a work directory that train.py may wipe


def prepare_workdir(path):
    """Create an empty work directory, replacing one from a previous run."""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise LeafError(f"{path}: exists and is not a directory")
        if os.listdir(path) and not os.path.exists(
                os.path.join(path, MARKER)):
            raise LeafError(f"{path}: exists and was not created by "
                            "train.py, choose another --workdir")
        shutil.rmtree(path)
    os.makedirs(path)
    open(os.path.join(path, MARKER), "w").close()


def save_model(workdir, model, meta):
    torch.save(model.state_dict(), os.path.join(workdir, MODEL))
    with open(os.path.join(workdir, META), "w") as f:
        json.dump(meta, f, indent=2)


def build_zip(workdir, zip_path):
    """Zip workdir under its own name; images are stored, not recompressed."""
    base = os.path.dirname(os.path.abspath(workdir))
    part = zip_path + ".part"
    with zipfile.ZipFile(part, "w") as zf:
        for dirpath, dirnames, files in os.walk(workdir):
            dirnames.sort()
            for name in sorted(f for f in files if not f.startswith(".")):
                path = os.path.join(dirpath, name)
                method = (zipfile.ZIP_STORED if is_image(name)
                          else zipfile.ZIP_DEFLATED)
                zf.write(path, os.path.relpath(path, base),
                         compress_type=method)
    os.replace(part, zip_path)


def sha1sum(path):
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_signature(zip_path, signature_path):
    """Write '<sha1>  <zip>' so `sha1sum -c signature.txt` can verify it."""
    folder = os.path.dirname(os.path.abspath(signature_path))
    digest = sha1sum(zip_path)
    with open(signature_path, "w") as f:
        f.write(f"{digest}  {os.path.relpath(zip_path, folder)}\n")
    return digest


def _read(path):
    if os.path.isdir(path):
        with open(os.path.join(path, META), "rb") as f:
            meta = f.read()
        with open(os.path.join(path, MODEL), "rb") as f:
            return meta, f.read()
    with zipfile.ZipFile(path) as zf:
        metas = sorted((n for n in zf.namelist()
                        if n.rsplit("/", 1)[-1] == META), key=len)
        if not metas:
            raise KeyError(f"no {META} inside the archive")
        root = metas[0][:-len(META)]
        return zf.read(root + META), zf.read(root + MODEL)


def load_learnings(path):
    """(model on CPU, meta) from learnings.zip or an extracted directory."""
    if not os.path.exists(path):
        raise LeafError(f"{path}: not found, run train.py first")
    if not os.path.isdir(path) and not zipfile.is_zipfile(path):
        raise LeafError(f"{path}: neither a zip archive nor a directory")
    try:
        meta_raw, model_raw = _read(path)
        meta = json.loads(meta_raw)
        meta["img_size"] = int(meta["img_size"])
        model = LeafCNN(len(meta["classes"]))
        state = torch.load(io.BytesIO(model_raw), map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state)
    except (OSError, KeyError, ValueError, TypeError, RuntimeError,
            EOFError, zipfile.BadZipFile, pickle.UnpicklingError) as err:
        raise LeafError(f"{path}: invalid learnings ({err})")
    return model.eval(), meta
