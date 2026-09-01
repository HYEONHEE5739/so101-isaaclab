#!/usr/bin/env python3
import argparse, h5py
'''
python scripts/tools/inspect_isaac_hdf5.py datasets/generated_demo_20ep_v3.hdf5
'''
parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

def walk(group, prefix=""):
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            print(f"[G] {path}")
            walk(value, path)
        else:
            print(f"[D] {path} shape={value.shape} dtype={value.dtype}")

with h5py.File(args.file, "r") as f:
    print(list(f["data"].keys()))
    print("root attrs:", dict(f.attrs))
    if "data" in f:
        print("data attrs:", dict(f["data"].attrs))
    walk(f)
