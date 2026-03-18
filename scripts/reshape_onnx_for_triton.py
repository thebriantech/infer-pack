#!/usr/bin/env python3
"""Wrap SCRFD ONNX outputs with reshape nodes so outputs have shape
[batch, anchors, channels].

Usage:
  python scripts/reshape_onnx_for_triton.py \
    --input models/face_detection_model/1/model.onnx \
    --output models/face_detection_model/1/model_batched.onnx

The script adds Reshape nodes that convert the original outputs
of shape [N, C] into [-1, anchors, C] (where -1 lets ONNX infer the
batch dimension). This makes the model outputs rank-3 and compatible
with a Triton config that uses `max_batch_size > 0` and explicit
per-stride dims (e.g. [12800,10]).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import onnx
from onnx import helper, numpy_helper, TensorProto


# Mapping from tensor name -> (anchors, channels)
OUTPUT_SHAPES: Dict[str, Tuple[int, int]] = {
    "448": (12800, 1),
    "471": (3200, 1),
    "494": (800, 1),
    "451": (12800, 4),
    "474": (3200, 4),
    "497": (800, 4),
    "454": (12800, 10),
    "477": (3200, 10),
    "500": (800, 10),
}


def make_shape_initializer(name: str, shape: List[int]) -> onnx.NodeProto:
    arr = numpy_helper.from_array(
        __import__("numpy").asarray(shape, dtype="int64"), name=name
    )
    return arr


def add_reshape_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph

    # Collect existing initializer names to avoid collisions
    existing_inits = {init.name for init in graph.initializer}

    new_nodes = []
    new_inits = []
    new_output_names = {}

    for out in list(graph.output):
        if out.name not in OUTPUT_SHAPES:
            continue

        anchors, channels = OUTPUT_SHAPES[out.name]

        # Create a unique initializer name for the reshape shape
        shape_name = f"shape_for_{out.name}"
        if shape_name in existing_inits:
            i = 1
            while f"{shape_name}_{i}" in existing_inits:
                i += 1
            shape_name = f"{shape_name}_{i}"

        # shape tensor: [-1, anchors, channels]
        shape_vals = [-1, anchors, channels]
        shape_init = make_shape_initializer(shape_name, shape_vals)
        new_inits.append(shape_init)

        # Rename any node outputs that equal the current output name to a temp name
        temp_name = f"{out.name}_orig"
        for node in graph.node:
            for idx, oname in enumerate(node.output):
                if oname == out.name:
                    node.output[idx] = temp_name
        # Also rename initializers / value infos if present
        for init in graph.initializer:
            if init.name == out.name:
                init.name = temp_name
        for value_info in list(graph.value_info):
            if value_info.name == out.name:
                value_info.name = temp_name

        # New reshape node: input is temp_name, output is original output name
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[temp_name, shape_name],
            outputs=[out.name],
            name=f"Reshape_{out.name}",
        )
        new_nodes.append(reshape_node)

    # Append new initializers and nodes to graph
    for init in new_inits:
        graph.initializer.append(init)
    for node in new_nodes:
        graph.node.append(node)

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    model = add_reshape_nodes(model)
    onnx.save(model, args.output)
    print(f"Saved reshaped model to: {args.output}")


if __name__ == "__main__":
    main()
