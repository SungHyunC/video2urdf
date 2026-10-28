"""Compiled decoder for SAM 3.1 one_bit masks (the "!" encoding).

A numba port of the one_bit path of meta-sam-parser 0.1.0 (_mask_codec._unpack and
_decode_raster), used under the SAM License; see LICENSES/SAM_LICENSE. The reference
decoder is pure Python and takes ~0.3 s per large mask, which makes a 600-frame track
take minutes. Canonical-form checks are skipped; scripts/check_fast_mask.py compares this
decoder against the reference on cached tracks.
"""

import numba
import numpy as np

_EXCLUDED = set('"\\,;<>|')
_ALPHABET = "".join(chr(c) for c in range(0x21, 0x7F) if chr(c) not in _EXCLUDED)
_VALUES = np.full(128, -1, np.int64)
for _i, _c in enumerate(_ALPHABET):
    _VALUES[ord(_c)] = _i

_M32 = 0xFFFFFFFF
_TOP = 1 << 24
_BOTTOM = 1 << 16
_INCREMENT = 14
_COUNT_LIMIT = 4096
_CONTEXTS = 1 << 12
_PAD = 2


def unpack(payload: str) -> np.ndarray:
    """Base85 payload (after the encoding marker) -> bytes, as uint8."""
    digits = _VALUES[np.frombuffer(payload.encode("ascii"), np.uint8)]
    if (digits < 0).any():
        raise ValueError("invalid character in mask payload")
    length = 0
    for d in digits[:5]:
        length = length * 85 + int(d)
    body = digits[5:]
    full, rem = divmod(length, 4)
    groups = body[: full * 5].reshape(-1, 5)
    values = (((groups[:, 0] * 85 + groups[:, 1]) * 85 + groups[:, 2]) * 85 + groups[:, 3]) * 85 + groups[:, 4]
    out = np.empty(length, np.uint8)
    out[: full * 4] = ((values[:, None] >> np.array([24, 16, 8, 0])) & 0xFF).reshape(-1)
    if rem:
        tail = list(body[full * 5:]) + [84] * (5 - len(body[full * 5:]))
        v = 0
        for d in tail:
            v = v * 85 + int(d)
        for i in range(rem):
            out[full * 4 + i] = (v >> (24 - 8 * i)) & 0xFF
    return out


@numba.njit(cache=True)
def _decode(data: np.ndarray, width: int, height: int) -> np.ndarray:
    n = data.shape[0]
    pos = 0
    code = np.uint64(0)
    for _ in range(4):
        b = data[pos] if pos < n else 0
        pos += 1
        code = ((code << np.uint64(8)) | np.uint64(b)) & np.uint64(_M32)
    low = np.uint64(0)
    rng = np.uint64(_M32)
    counts = np.ones(_CONTEXTS * 2, np.int64)
    pw = width + 2 * _PAD
    padded = np.zeros(pw * (height + _PAD), np.uint8)
    out = np.zeros(width * height, np.uint8)
    for y in range(height):
        row = (y + _PAD) * pw + _PAD
        for x in range(width):
            p = row + x
            q = p - pw
            e = p - 2 * pw
            ctx = (padded[p - 1] | (padded[p - 2] << 1) | (padded[q - 2] << 2) | (padded[q - 1] << 3)
                   | (padded[q] << 4) | (padded[q + 1] << 5) | (padded[q + 2] << 6) | (padded[e - 2] << 7)
                   | (padded[e - 1] << 8) | (padded[e] << 9) | (padded[e + 1] << 10) | (padded[e + 2] << 11))
            off = ctx << 1
            zero = counts[off]
            one = counts[off + 1]
            total = np.uint64(zero + one)
            scaled = rng // total
            value = ((code - low) & np.uint64(_M32)) // scaled
            if value >= total:
                value = total - np.uint64(1)
            if value < np.uint64(zero):
                bit = 0
                cum, freq = np.uint64(0), np.uint64(zero)
            else:
                bit = 1
                cum, freq = np.uint64(zero), np.uint64(one)
            low = (low + scaled * cum) & np.uint64(_M32)
            rng = scaled * freq
            while True:
                if (low ^ ((low + rng) & np.uint64(_M32))) < np.uint64(_TOP):
                    pass
                elif rng < np.uint64(_BOTTOM):
                    rng = ((np.uint64(0) - low) & np.uint64(_M32)) & np.uint64(_BOTTOM - 1)
                else:
                    break
                b = data[pos] if pos < n else 0
                pos += 1
                code = ((code << np.uint64(8)) | np.uint64(b)) & np.uint64(_M32)
                low = (low << np.uint64(8)) & np.uint64(_M32)
                rng = (rng << np.uint64(8)) & np.uint64(_M32)
            padded[p] = bit
            out[y * width + x] = bit
            c = (counts[off + bit] + _INCREMENT) & 0xFFFF
            counts[off + bit] = c
            if counts[off] + counts[off + 1] >= _COUNT_LIMIT:
                counts[off] = max(counts[off] >> 1, 1)
                counts[off + 1] = max(counts[off + 1] >> 1, 1)
    return out


def decode_one_bit(payload: str, width: int, height: int) -> np.ndarray:
    """``payload`` starts with the "!" marker. Returns a (height, width) bool raster."""
    return _decode(unpack(payload[1:]), width, height).reshape(height, width).astype(bool)
