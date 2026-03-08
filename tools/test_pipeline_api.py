#!/usr/bin/env python3
"""Quick tester for InferPack HTTP / gRPC pipeline APIs.

Examples:
  # HTTP
  python tools/test_pipeline_api.py \
    --protocol http \
    --face-detection /path/to/image.jpg

  python tools/test_pipeline_api.py \
    --protocol http \
    --face-comparison /path/to/image1.jpg /path/to/image2.jpg

  # gRPC (requires generated stubs)
  python scripts/gen_proto.sh
  python tools/test_pipeline_api.py \
    --protocol grpc \
    --face-comparison /path/to/image1.jpg /path/to/image2.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx


def _read_bytes(path: str) -> bytes:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return p.read_bytes()


def _print(title: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_http(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.http_port}"

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        if args.face_detection:
            image_path = args.face_detection
            with open(image_path, "rb") as fh:
                response = client.post(
                    f"{base_url}/api/v1/pipelines/face_detection",
                    files={"image": (Path(image_path).name, fh, "application/octet-stream")},
                )
            response.raise_for_status()
            _print("HTTP face_detection", response.json())

        if args.face_comparison:
            image1, image2 = args.face_comparison
            with open(image1, "rb") as fh1, open(image2, "rb") as fh2:
                response = client.post(
                    f"{base_url}/api/v1/pipelines/face_comparison",
                    files={
                        "image_1": (Path(image1).name, fh1, "application/octet-stream"),
                        "image_2": (Path(image2).name, fh2, "application/octet-stream"),
                    },
                )
            response.raise_for_status()
            _print("HTTP face_comparison", response.json())


def run_grpc(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent.parent
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        import grpc
        from inferpack.api.grpc import inferpack_pb2 as pb2
        from inferpack.api.grpc import inferpack_pb2_grpc as pb2_grpc
    except ImportError as exc:
        raise RuntimeError(
            "gRPC stubs are missing or grpc package unavailable. "
            "Run: python scripts/gen_proto.sh and ensure grpcio is installed."
        ) from exc

    target = f"{args.host}:{args.grpc_port}"
    with grpc.insecure_channel(target) as channel:
        stub = pb2_grpc.PipelineServiceStub(channel)

        if args.face_detection:
            payload = _read_bytes(args.face_detection)
            res = stub.ExecuteFaceDetection(pb2.ExecuteFaceDetectionRequest(image=payload))
            _print(
                "gRPC face_detection",
                {
                    "pipeline_name": res.pipeline_name,
                    "success": res.success,
                    "error": res.error,
                    "output": json.loads(res.output_json or "{}"),
                },
            )

        if args.face_comparison:
            image1, image2 = args.face_comparison
            res = stub.ExecuteFaceComparison(
                pb2.ExecuteFaceComparisonRequest(
                    image_1=_read_bytes(image1),
                    image_2=_read_bytes(image2),
                )
            )
            _print(
                "gRPC face_comparison",
                {
                    "pipeline_name": res.pipeline_name,
                    "success": res.success,
                    "error": res.error,
                    "output": json.loads(res.output_json or "{}"),
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test InferPack HTTP/gRPC pipeline APIs")
    parser.add_argument("--protocol", choices=["http", "grpc"], required=True)
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--http-port", type=int, default=29000, help="HTTP port (default: 29000)")
    parser.add_argument("--grpc-port", type=int, default=29001, help="gRPC port (default: 29001)")
    parser.add_argument(
        "--face-detection",
        metavar="IMAGE",
        help="Run face_detection endpoint/RPC with one image",
    )
    parser.add_argument(
        "--face-comparison",
        nargs=2,
        metavar=("IMAGE_1", "IMAGE_2"),
        help="Run face_comparison endpoint/RPC with two images",
    )

    args = parser.parse_args()
    if not args.face_detection and not args.face_comparison:
        parser.error("At least one of --face-detection or --face-comparison is required.")
    return args


def main() -> None:
    args = parse_args()
    if args.protocol == "http":
        run_http(args)
    else:
        run_grpc(args)


if __name__ == "__main__":
    main()
