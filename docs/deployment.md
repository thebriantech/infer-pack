# Deployment

## 1. Manage built-in models

Built-in ONNX models are hosted on HuggingFace ([bangpc/infer-pack](https://huggingface.co/bangpc/infer-pack)).
`scripts/download_models.py` handles both downloading and uploading them.

### Download (first-time setup)

Pull models from HuggingFace into `models/`:

```bash
make download-models
# equivalent: python scripts/download_models.py download
```

Pass `--force` to re-download files that already exist:

```bash
python scripts/download_models.py download --force
```

The script places files at:

```
models/
  face_detection_model/
    config.pbtxt
    1/model.onnx          # SCRFD-10G  (~17 MB)
  face_feature_extraction_model/
    config.pbtxt
    1/model.onnx          # ArcFace-R50 (~167 MB)
```

### Upload (publish updated models)

After modifying or replacing model files locally, push them back to HuggingFace:

```bash
# log in once (stores a token in ~/.cache/huggingface/)
huggingface-cli login

make upload-models
# equivalent: python scripts/download_models.py upload
```

Supply a custom commit message if needed:

```bash
python scripts/download_models.py upload --message "Retrain ArcFace on extended dataset"
```

The upload will fail with a clear error if any of the expected local files are missing.

## 2. Docker Compose

```bash
make compose-up
```

Ports are configured in:

- `docker/docker-compose.yaml`
- `config/inferpack.yaml`

## Local development

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m inferpack.main
```

## API test helper

```bash
python tools/test_pipeline_api.py --help
```
