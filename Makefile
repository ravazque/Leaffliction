PYTHON		?= python3.13
VENV		= .venv
PY			= $(VENV)/bin/python
STAMP		= $(VENV)/.installed

# cpu: CPU build of PyTorch (about 1.5 GB); cuda: the default CUDA build (about 6 GB)
TORCH		?= cpu
TORCH_INDEX	= $(if $(filter cpu,$(TORCH)),--extra-index-url https://download.pytorch.org/whl/cpu)

LEARNINGS_URL	= https://github.com/ravazque/Leaffliction/releases/download/v1.0/learnings.zip
VALIDATION		= learnings/validation
IMAGES			?= images
IMG				?=


help:
	@echo "make install           .venv with Python 3.13 and requirements.txt (TORCH=cpu|cuda)"
	@echo "make learnings         download learnings.zip and check it against signature.txt"
	@echo "make predict           classify one leaf, then score the validation images (IMG=path for one image)"
	@echo "make distribution      class counts and charts of images/"
	@echo "make augmentation      balanced copy of images/ in augmented_directory/"
	@echo "make transformation    PlantCV views of the first image of images/"
	@echo "make train             train on images/, writes learnings.zip and signature.txt"
	@echo "make test              pytest suite"
	@echo "make lint              flake8"
	@echo "make clean             generated figures and folders"
	@echo "make fclean            clean, the .venv and learnings/ (learnings.zip is kept)"

install: $(STAMP)

$(STAMP): requirements.txt
	@if command -v $(PYTHON) >/dev/null; then $(PYTHON) -m venv $(VENV); \
	elif command -v uv >/dev/null; then uv venv -q --python 3.13 --seed $(VENV); \
	else echo "Python 3.13 not found: install it, or install uv (https://docs.astral.sh/uv/)" >&2; exit 1; fi
	$(PY) -m pip install -q -r requirements.txt $(TORCH_INDEX)
	@touch $@

learnings.zip:
	@command -v curl >/dev/null || { echo "curl not found: install curl" >&2; exit 1; }
	curl -fL --retry 3 -o $@.part $(LEARNINGS_URL)
	mv $@.part $@

learnings: learnings.zip
	@if command -v sha1sum >/dev/null; then sha1sum -c signature.txt; else shasum -c signature.txt; fi

$(VALIDATION): learnings.zip
	unzip -q -o learnings.zip '$(VALIDATION)/*'
	@touch $@

predict: $(STAMP) learnings $(VALIDATION)
ifeq ($(IMG),)
	$(PY) src/predict.py "$$(find $(VALIDATION)/Grape_spot -type f | sort | head -1)"
	$(PY) src/predict.py $(VALIDATION)
else
	$(PY) src/predict.py "$(IMG)"
endif

images-check:
	@[ -d $(IMAGES) ] || { echo "$(IMAGES)/ not found: extract the leaf dataset at the repository root" >&2; exit 1; }

distribution: $(STAMP) images-check
	$(PY) src/Distribution.py $(IMAGES)

augmentation: $(STAMP) images-check
	$(PY) src/Augmentation.py $(IMAGES)

transformation: $(STAMP) images-check
	$(PY) src/Transformation.py "$$(find $(IMAGES) -type f -iname '*.jpg' | sort | head -1)"

train: $(STAMP) images-check
	$(PY) src/train.py $(IMAGES)

test: $(STAMP)
	$(PY) -m pytest -q

lint: $(STAMP)
	$(PY) -m flake8 src tests

clean:
	rm -rf augmented_directory transformed .pytest_cache
	rm -f *_distribution.png augmentation.png transformations.png histogram.png prediction.png
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +

# learnings.zip is kept: it is the signed archive
fclean: clean
	rm -rf $(VENV) learnings

re: fclean install

.PHONY: help install learnings predict images-check distribution augmentation transformation \
		train test lint clean fclean re
