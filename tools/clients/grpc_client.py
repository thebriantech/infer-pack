#!/usr/bin/env python3
"""
gRPC client for InferPack pipeline APIs.
Supports: list, activate, deactivate, face_detection, face_comparison
"""
import argparse
import json
import sys
from pathlib import Path

def _print(title, payload):
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

def run(args):
    root = Path(__file__).resolve().parent.parent.parent
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
        if args.list:
            resp = stub.ListPipelines(pb2.ListPipelinesRequest())
            from google.protobuf.json_format import MessageToDict
            _print("gRPC list_pipelines", {"pipelines": [MessageToDict(p) for p in resp.pipelines]})
        if args.activate:
            resp = stub.ActivatePipeline(pb2.PipelineRequest(name=args.activate))
            _print("gRPC activate_pipeline", {"status": resp.status, "pipeline": resp.pipeline})
        if args.deactivate:
            resp = stub.DeactivatePipeline(pb2.PipelineRequest(name=args.deactivate))
            _print("gRPC deactivate_pipeline", {"status": resp.status, "pipeline": resp.pipeline})
        if args.face_detection:
            with open(args.face_detection, "rb") as fh:
                img_bytes = fh.read()
            req = pb2.ExecuteFaceDetectionRequest(image=img_bytes)
            resp = stub.ExecuteFaceDetection(req)
            _print("gRPC face_detection", json.loads(resp.output_json or "{}"))
        if args.face_comparison:
            img1, img2 = args.face_comparison
            with open(img1, "rb") as fh1, open(img2, "rb") as fh2:
                req = pb2.ExecuteFaceComparisonRequest(
                    image_1=fh1.read(),
                    image_2=fh2.read()
                )
            resp = stub.ExecuteFaceComparison(req)
            _print("gRPC face_comparison", json.loads(resp.output_json or "{}"))

def parse_args():
    p = argparse.ArgumentParser(description="InferPack gRPC API client")
    p.add_argument("--host", default="localhost")
    p.add_argument("--grpc-port", type=int, default=29001)
    p.add_argument("--list", action="store_true", help="List pipelines")
    p.add_argument("--activate", metavar="NAME", help="Activate pipeline by name")
    p.add_argument("--deactivate", metavar="NAME", help="Deactivate pipeline by name")
    p.add_argument("--face-detection", metavar="IMAGE", help="Run face_detection with one image")
    p.add_argument("--face-comparison", nargs=2, metavar=("IMAGE_1", "IMAGE_2"), help="Run face_comparison with two images")
    args = p.parse_args()
    if not (args.list or args.activate or args.deactivate or args.face_detection or args.face_comparison):
        p.error("At least one action is required.")
    return args

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()
