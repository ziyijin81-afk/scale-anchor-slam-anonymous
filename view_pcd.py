#!/usr/bin/env python3
"""Serve a binary PCD point cloud in a browser with Viser.

The file is memory-mapped and evenly sampled, so a large PCD does not need to
be copied into RAM in full. The source file is never modified.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import viser
except ImportError as exc:
    raise SystemExit(
        "Viser is not installed in this Python environment. Run with:\n"
        "  conda run -n vggt-slam python view_pcd.py <file.pcd>"
    ) from exc


def read_header(path: Path) -> tuple[dict[str, list[str]], int]:
    """Return parsed PCD header fields and byte offset of binary data."""
    header: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PCD header ended before a DATA line")
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("Invalid PCD header") from exc
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                return header, stream.tell()


def pcd_dtype(header: dict[str, list[str]]) -> np.dtype:
    fields = header["FIELDS"]
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("Inconsistent FIELDS/SIZE/TYPE/COUNT entries")

    type_codes = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    dtype_fields = []
    for name, size, kind, count in zip(fields, sizes, types, counts):
        try:
            dtype = type_codes[(kind.upper(), size)]
        except KeyError as exc:
            raise ValueError(f"Unsupported PCD field type: {kind}{size}") from exc
        dtype_fields.append((name, dtype) if count == 1 else (name, dtype, (count,)))
    return np.dtype(dtype_fields)


def load_sample(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    header, offset = read_header(path)
    if header.get("DATA", [""])[0].lower() != "binary":
        raise ValueError("Only uncompressed binary PCD files are supported")

    dtype = pcd_dtype(header)
    total = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    if total <= 0:
        raise ValueError("PCD contains no points")
    cloud = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(total,))
    if not {"x", "y", "z"}.issubset(dtype.names or ()):
        raise ValueError("PCD must contain x, y, and z fields")

    # Even stride sampling is deterministic and avoids allocating an index
    # array proportional to the full (potentially very large) point cloud.
    stride = max(1, int(np.ceil(total / max_points)))
    sample = cloud[::stride]
    points = np.column_stack((sample["x"], sample["y"], sample["z"])).astype(
        np.float32, copy=False
    )

    names = dtype.names or ()
    if "rgb" in names or "rgba" in names:
        packed = np.asarray(sample["rgb" if "rgb" in names else "rgba"])
        if packed.dtype.kind == "f":
            packed = packed.view(np.dtype(f"<u{packed.dtype.itemsize}"))
        else:
            packed = packed.astype(np.uint32, copy=False)
        colors = np.column_stack(
            ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
        ).astype(np.uint8)
    elif {"r", "g", "b"}.issubset(names):
        colors = np.column_stack((sample["r"], sample["g"], sample["b"])).astype(
            np.uint8
        )
    else:
        colors = np.full((len(points), 3), 200, dtype=np.uint8)

    valid = np.isfinite(points).all(axis=1)
    return points[valid], colors[valid], total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd", type=Path, help="Binary PCD file to view")
    parser.add_argument("--port", type=int, default=8081, help="HTTP port (default: 8081)")
    parser.add_argument(
        "--max-points",
        type=int,
        default=750_000,
        help="Maximum points sent to each browser (default: 750000)",
    )
    parser.add_argument(
        "--point-size", type=float, default=0.003, help="Rendered point size"
    )
    args = parser.parse_args()
    if args.max_points <= 0:
        parser.error("--max-points must be positive")
    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    path = args.pcd.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"PCD file not found: {path}")

    print(f"Reading a display sample from {path}", flush=True)
    try:
        points, colors, total = load_sample(path, args.max_points)
    except (KeyError, ValueError, OSError) as exc:
        raise SystemExit(f"Could not load PCD: {exc}") from exc
    print(f"Displaying {len(points):,} of {total:,} points", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown(
        f"**{path.name}**  \nDisplaying {len(points):,} / {total:,} points"
    )
    server.scene.add_point_cloud(
        "/point_cloud",
        points=points,
        colors=colors,
        point_size=args.point_size,
        point_shape="circle",
    )
    server.scene.add_grid(
        "/ground",
        width=10.0,
        height=10.0,
        position=(0.0, 0.0, 0.0),
    )
    print(f"Point-cloud viewer ready at http://localhost:{args.port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping viewer.", flush=True)


if __name__ == "__main__":
    main()
