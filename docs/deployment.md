# Deployment

## Docker Compose

```bash
cd docker
docker compose up -d --build
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
