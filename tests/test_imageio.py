import cv2
import numpy as np
import pytest
from conftest import leaf_image, write

from leaffliction.cli import LeafError
from leaffliction.imageio import load_image, save_image, to_rgb


def test_grayscale_and_alpha_images_load_as_bgr(tmp_path):
    gray = np.full((20, 30), 90, np.uint8)
    rgba = np.zeros((20, 30, 4), np.uint8)
    for name, img in (("g.png", gray), ("a.png", rgba)):
        assert load_image(write(str(tmp_path / name), img)).shape == (20,
                                                                      30, 3)


def test_unicode_and_spaces_in_path_round_trip(tmp_path):
    path = str(tmp_path / "hojas ñ" / "image (1).JPG")
    write(path, leaf_image()[0])
    img = load_image(path)
    save_image(path.replace(".JPG", ".png"), img)
    assert np.array_equal(load_image(path.replace(".JPG", ".png")), img)


@pytest.mark.parametrize("content", [b"", b"plain text", b"\xff\xd8\xff"])
def test_unreadable_files_raise_leaf_error(tmp_path, content):
    path = tmp_path / "broken.JPG"
    path.write_bytes(content)
    with pytest.raises(LeafError, match="not a readable image"):
        load_image(str(path))


def test_truncated_jpeg_never_crashes(tmp_path):
    ok, buf = cv2.imencode(".jpg", leaf_image()[0])
    path = tmp_path / "cut.JPG"
    path.write_bytes(buf.tobytes()[:200])
    try:
        assert load_image(str(path)).ndim == 3
    except LeafError:
        pass


def test_directory_and_missing_paths(tmp_path):
    with pytest.raises(LeafError, match="directory"):
        load_image(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "missing.JPG"))


def test_unknown_extension_cannot_be_saved(tmp_path):
    with pytest.raises(LeafError, match="cannot encode"):
        save_image(str(tmp_path / "x.unknown"), leaf_image()[0])


def test_to_rgb_swaps_channels():
    img = np.zeros((1, 1, 3), np.uint8)
    img[0, 0] = (1, 2, 3)
    assert to_rgb(img)[0, 0].tolist() == [3, 2, 1]
    assert to_rgb(np.zeros((2, 2), np.uint8)).shape == (2, 2, 3)
