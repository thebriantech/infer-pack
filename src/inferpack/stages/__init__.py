"""Pipeline stage implementations — organised by domain.

Layout::

    stages/
    ├── _utils/              Internal helper functions (NMS, image ops)
    ├── shared/              Domain-agnostic reusable stages
    │   ├── image_preprocess
    │   ├── region_crop
    │   ├── detection_postprocess
    │   └── feature_comparison
    ├── face/                Face detection & recognition stages
    │   ├── scrfd_detection
    │   └── arcface_extraction
    └── registry.py          Auto-discovery stage registry

To add a new domain, create ``stages/<domain>/__init__.py`` with::

    def get_stages(inference_client, model_registry) -> dict[str, PipelineStage]:
        ...

and register the package path in ``registry._DOMAIN_PACKAGES``.
"""
