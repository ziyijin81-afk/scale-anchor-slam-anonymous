"""Pure helpers for forming fixed-size offline submap batches."""


def should_process_submap(buffer_size, target_size, overlap_size, processed_submaps, is_last):
    """Return whether a full batch or a non-empty final tail should be processed."""
    buffer_size = int(buffer_size)
    target_size = int(target_size)
    overlap_size = int(overlap_size)
    processed_submaps = int(processed_submaps)
    if target_size <= 0 or overlap_size < 0 or overlap_size >= target_size:
        raise ValueError("Require 0 <= overlap_size < target_size")
    if buffer_size >= target_size:
        return True
    if not is_last:
        return False
    carried_overlap = overlap_size if processed_submaps > 0 else 0
    return buffer_size > carried_overlap


def pad_submap_frames(frame_names, target_size):
    """Repeat the final selected frame until a short tail reaches target_size."""
    frames = list(frame_names)
    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if not frames:
        raise ValueError("cannot pad an empty submap")
    if len(frames) > target_size:
        raise ValueError("submap already exceeds target_size")
    padding_count = target_size - len(frames)
    if padding_count:
        frames.extend([frames[-1]] * padding_count)
    return frames, padding_count
