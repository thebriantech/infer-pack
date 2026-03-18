#!/usr/bin/env python3
"""Rename ONNX model outputs ending with '_reshaped' to remove the suffix.

Usage: python scripts/rename_onnx_outputs.py <model_path>
If no path given, defaults to models/face_detection_model/1/model.onnx
"""
import sys
import os

def main():
    try:
        import onnx
    except Exception as e:
        print('onnx not installed:', e)
        print('Install with: pip install onnx')
        return 2

    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/face_detection_model/1/model.onnx'
    if not os.path.exists(model_path):
        print('Model path not found:', model_path)
        return 3

    out_path = model_path
    backup_path = model_path + '.bak'
    if not os.path.exists(backup_path):
        os.rename(model_path, backup_path)
        print('Backed up original to', backup_path)
    else:
        print('Backup already exists at', backup_path, '- overwriting output will proceed')

    model = onnx.load(backup_path)
    mapping = {}

    for out in model.graph.output:
        if out.name.endswith('_reshaped'):
            new_name = out.name.replace('_reshaped', '')
            mapping[out.name] = new_name
            out.name = new_name

    # Update value_info
    for vi in model.graph.value_info:
        if vi.name in mapping:
            vi.name = mapping[vi.name]

    # Update inputs if needed
    for inp in model.graph.input:
        if inp.name in mapping:
            inp.name = mapping[inp.name]

    # Update initializers
    for init in model.graph.initializer:
        if init.name in mapping:
            init.name = mapping[init.name]

    # Update node inputs/outputs
    for node in model.graph.node:
        for i, n in enumerate(node.output):
            if n in mapping:
                node.output[i] = mapping[n]
        for i, n in enumerate(node.input):
            if n in mapping:
                node.input[i] = mapping[n]

    onnx.save(model, out_path)
    print('Saved renamed model to', out_path)
    print('Renamed entries:', mapping)
    return 0

if __name__ == '__main__':
    sys.exit(main())
