#!/usr/bin/env python3
"""
HTTP client for InferPack pipeline APIs.
Supports: list, activate, deactivate, face_detection, face_comparison
"""
import argparse
import json
from pathlib import Path
import sys
import httpx

def _print(title, payload):
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

def run(args):
    base_url = f"http://{args.host}:{args.http_port}/api/v1/pipelines"
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        if args.list:
            resp = client.get(base_url)
            resp.raise_for_status()
            _print("HTTP list_pipelines", resp.json())
        if args.activate:
            resp = client.post(f"{base_url}/{args.activate}/activate")
            resp.raise_for_status()
            _print("HTTP activate_pipeline", resp.json())
        if args.deactivate:
            resp = client.post(f"{base_url}/{args.deactivate}/deactivate")
            resp.raise_for_status()
            _print("HTTP deactivate_pipeline", resp.json())
        if args.face_detection:
            with open(args.face_detection, "rb") as fh:
                resp = client.post(f"{base_url}/face_detection", files={"image": (Path(args.face_detection).name, fh, "application/octet-stream")})
            resp.raise_for_status()
            _print("HTTP face_detection", resp.json())
        if args.face_comparison:
            img1, img2 = args.face_comparison
            with open(img1, "rb") as fh1, open(img2, "rb") as fh2:
                resp = client.post(f"{base_url}/face_comparison", files={
                    "image_1": (Path(img1).name, fh1, "application/octet-stream"),
                    "image_2": (Path(img2).name, fh2, "application/octet-stream")
                })
            resp.raise_for_status()
            _print("HTTP face_comparison", resp.json())

def parse_args():
    p = argparse.ArgumentParser(description="InferPack HTTP API client")
    p.add_argument("--host", default="localhost")
    p.add_argument("--http-port", type=int, default=29000)
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
