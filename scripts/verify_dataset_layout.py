#!/usr/bin/env python3
"""Check that the prepared domain tensors match the loader's expected layout."""

import argparse
from pathlib import Path

import numpy as np
import torch


SPECS = {
    "WISDM": {"domains": range(36), "channels": 3, "length": 128, "classes": 6},
    "FD": {"domains": range(4), "channels": 1, "length": 5120, "classes": 3},
    "Boiler": {"domains": range(1, 4), "channels": 20, "length": 36, "classes": 2},
}


def to_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def check_file(path, spec):
    # These are trusted local dataset files; the repository's loader uses the same pickle format.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"samples", "labels"}.issubset(payload):
        raise ValueError("expected a dictionary containing 'samples' and 'labels'")

    samples = to_array(payload["samples"])
    labels = to_array(payload["labels"]).reshape(-1)
    if samples.ndim == 2:
        valid_shape = spec["channels"] == 1 and samples.shape[1] == spec["length"]
    elif samples.ndim == 3:
        valid_shape = samples.shape[1:] in (
            (spec["length"], spec["channels"]),
            (spec["channels"], spec["length"]),
        )
    else:
        valid_shape = False

    if not valid_shape:
        raise ValueError(
            f"unexpected samples shape {samples.shape}; expected N x {spec['length']} x {spec['channels']}"
        )
    if samples.shape[0] != labels.shape[0]:
        raise ValueError(f"sample/label count mismatch: {samples.shape[0]} vs {labels.shape[0]}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must use an integer dtype, got {labels.dtype}")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= spec["classes"]:
        raise ValueError(f"labels must be non-empty integers in [0, {spec['classes'] - 1}]")
    return samples.shape, labels.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset", choices=["WISDM", "FD", "Boiler"])
    args = parser.parse_args()

    names = [args.dataset] if args.dataset else list(SPECS)
    errors = []
    checked = 0
    for name in names:
        spec = SPECS[name]
        for domain in spec["domains"]:
            for split in ("train", "test"):
                path = args.data_root / name / f"{split}_{domain}.pt"
                try:
                    if not path.is_file():
                        raise FileNotFoundError("file is missing")
                    shape, count = check_file(path, spec)
                    print(f"OK {path}: samples={count}, shape={shape}")
                    checked += 1
                except Exception as exc:  # Report every missing or invalid split in one pass.
                    errors.append(f"{path}: {exc}")

    if errors:
        print("\nDataset validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"\nValidated {checked} files for {', '.join(names)}.")


if __name__ == "__main__":
    main()
