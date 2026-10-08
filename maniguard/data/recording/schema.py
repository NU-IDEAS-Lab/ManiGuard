"""Source episode schema and non-executable binary array serialization."""

from __future__ import annotations

import json
import zlib
from io import BytesIO

import msgpack
import numpy as np
from PIL import Image

SCHEMA_VERSION = 1


def _pack_default(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be serialized")
        return msgpack.ExtType(1, msgpack.packb((value.dtype.str, value.shape, value.tobytes()), use_bin_type=True))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return msgpack.ExtType(
            2, msgpack.packb(list(value), default=_pack_default, strict_types=True, use_bin_type=True)
        )
    raise TypeError(f"Unsupported record value: {type(value).__name__}")


def _unpack_ext(code, data):
    if code == 1:
        dtype, shape, raw = msgpack.unpackb(data, raw=False)
        dtype = np.dtype(dtype)
        if dtype.hasobject or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid serialized array")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if code == 2:
        return tuple(msgpack.unpackb(data, raw=False, ext_hook=_unpack_ext, strict_map_key=False))
    raise ValueError(f"Unknown binary record extension: {code}")


def pack_record(record) -> bytes:
    return zlib.compress(msgpack.packb(record, default=_pack_default, strict_types=True, use_bin_type=True), level=1)


def unpack_record(payload: bytes):
    return msgpack.unpackb(zlib.decompress(payload), raw=False, ext_hook=_unpack_ext, strict_map_key=False)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def json_text(value):
    return json.dumps(value, default=json_default, allow_nan=False, ensure_ascii=False)


def validate_metadata(metadata: dict) -> dict:
    result = json.loads(json_text(metadata))
    keys = result.get("camera_keys")
    if (
        not keys
        or len(set(keys)) != len(keys)
        or any(not isinstance(k, str) or "/" in k or k in (".", "..") for k in keys)
    ):
        raise ValueError("camera_keys must contain unique simple names")
    resolution = result.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(x) is not int or x <= 0 for x in resolution)
    ):
        raise ValueError("resolution must contain positive height and width")
    enc = result.get("rgb_encoding", {})
    if enc.get("codec") != "jpeg" or enc.get("subsampling") != "444" or enc.get("color_order") != "RGB":
        raise ValueError("Expected JPEG 444 RGB encoding")
    if type(enc.get("quality")) is not int or not 1 <= enc["quality"] <= 100:
        raise ValueError("Invalid JPEG quality")
    result["schema_version"] = SCHEMA_VERSION
    return result


def encode_rgb(rgb, metadata: dict) -> bytes:
    rgb = np.asarray(rgb)
    expected = (*metadata["resolution"], 3)
    if rgb.shape != expected or rgb.dtype != np.uint8:
        raise ValueError(f"Expected RGB uint8 {expected}, got {rgb.dtype} {rgb.shape}")
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=metadata["rgb_encoding"]["quality"], subsampling=0)
    return stream.getvalue()


def decode_rgb(payload: bytes) -> np.ndarray:
    if not payload.endswith(b"\xff\xd9"):
        raise ValueError("Truncated source JPEG")
    with Image.open(BytesIO(payload)) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError("Source JPEG is not RGB")
        return np.asarray(image).copy()


def append_blob(group, name: str, payload: bytes):
    if name not in group:
        target = group.create_group(name)
        target.create_dataset("bytes", shape=(0,), maxshape=(None,), dtype="u1", chunks=(65536,))
        target.create_dataset("offsets", data=np.array([0], dtype="i8"), maxshape=(None,), chunks=(256,))
    target = group[name]
    data, offsets = target["bytes"], target["offsets"]
    end = len(data) + len(payload)
    data.resize((end,))
    data[end - len(payload) : end] = np.frombuffer(payload, dtype="u1")
    offsets.resize((len(offsets) + 1,))
    offsets[-1] = end


def read_blob(group, index: int) -> bytes:
    offsets = group["offsets"]
    if index < 0 or index >= len(offsets) - 1:
        raise IndexError(index)
    start, end = int(offsets[index]), int(offsets[index + 1])
    if start < 0 or end <= start or end > len(group["bytes"]):
        raise ValueError("Invalid blob offsets")
    return group["bytes"][start:end].tobytes()


def validate_blobs(group, count: int):
    offsets = group["offsets"][:]
    if (
        len(offsets) != count + 1
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or offsets[-1] != len(group["bytes"])
    ):
        raise ValueError(f"Invalid offsets for {group.name}")
