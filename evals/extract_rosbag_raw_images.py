#!/usr/bin/env python3
"""Extract a bounded number of ROS Image messages as timestamped JPEG files."""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--timestamps-from-run-log",
        type=Path,
        default=None,
        help="Only extract bag-record timestamps referenced as image names in this log",
    )
    parser.add_argument(
        "--preserve-rgb-buffer-order",
        action="store_true",
        help="Write rgb8 buffers through OpenCV without RGB-to-BGR conversion",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    requested = None
    if args.timestamps_from_run_log is not None:
        requested = {
            int(value)
            for value in re.findall(
                r"(?<!\d)(\d{16,20})\.(?:jpg|jpeg|png)",
                args.timestamps_from_run_log.read_text(errors="replace"),
                flags=re.IGNORECASE,
            )
        }
        if not requested:
            raise ValueError(f"No image timestamps found in {args.timestamps_from_run_log}")

    count = 0
    with AnyReader([args.bag]) as reader:
        connections = [c for c in reader.connections if c.topic == args.topic]
        if not connections:
            raise ValueError(f"Topic not found: {args.topic}")
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if requested is not None and int(timestamp) not in requested:
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            encoding = str(msg.encoding).lower()
            channels = 1 if encoding in {"mono8", "8uc1"} else 3
            image = np.asarray(msg.data, dtype=np.uint8).reshape(
                int(msg.height), int(msg.step)
            )[:, : int(msg.width) * channels]
            image = image.reshape(int(msg.height), int(msg.width), channels)
            if encoding == "rgb8" and not args.preserve_rgb_buffer_order:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif encoding not in {"rgb8", "bgr8", "mono8", "8uc1"}:
                raise ValueError(f"Unsupported image encoding: {msg.encoding}")
            stamp = (
                int(timestamp)
                if requested is not None
                else int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            )
            path = args.output / f"{stamp}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Failed to write {path}")
            count += 1
            if count % 500 == 0:
                print(f"Extracted {count} frames", flush=True)
            if args.max_frames is not None and count >= args.max_frames:
                break
            if requested is not None and count == len(requested):
                break
    if requested is not None and count != len(requested):
        raise RuntimeError(f"Found {count} of {len(requested)} requested timestamps")
    print(f"Extracted total: {count}")


if __name__ == "__main__":
    main()
