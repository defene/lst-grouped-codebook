"""Independently decodable, byte-exact Prithvi sample frames."""
import hashlib
import struct
import numpy as np
import zstandard as zstd

MAGIC = b'EOZ1'
FILENAMES = ('bands.npy', 'fmask.npy', 'lst.npy', 'metadata.json')

def _encode_words(raw):
    if len(raw) % 2:
        raise ValueError('Word transform requires even byte count')
    words = np.frombuffer(raw, dtype='<u2')
    delta = np.empty_like(words)
    delta[:1] = words[:1]
    np.subtract(words[1:], words[:-1], out=delta[1:])
    return delta.view(np.uint8).reshape(-1, 2).T.copy().tobytes()

def _decode_words(raw):
    delta = np.frombuffer(raw, dtype=np.uint8).reshape(2, -1).T.copy().view('<u2').reshape(-1)
    return np.cumsum(delta, dtype=np.uint16).astype('<u2', copy=False).tobytes()

def original_digest(parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(struct.pack('<Q', len(part)))
        h.update(part)
    return h.hexdigest()

def encode(parts, level=3):
    if len(parts) != 4:
        raise ValueError('Expected four original files')
    payload = MAGIC + struct.pack('<4I', *(len(p) for p in parts))
    payload += _encode_words(parts[0]) + parts[1] + _encode_words(parts[2]) + parts[3]
    return zstd.ZstdCompressor(level=level, write_checksum=True).compress(payload)

def decode(frame, expected_digest=None):
    raw = zstd.ZstdDecompressor().decompress(frame, max_output_size=16 * 1024 * 1024)
    if raw[:4] != MAGIC:
        raise ValueError('Unsupported sample frame')
    lengths = struct.unpack('<4I', raw[4:20])
    if sum(lengths) + 20 != len(raw):
        raise ValueError('Invalid sample lengths')
    parts = []
    offset = 20
    for i, n in enumerate(lengths):
        part = raw[offset:offset+n]
        parts.append(_decode_words(part) if i in (0, 2) else part)
        offset += n
    if expected_digest and original_digest(parts) != expected_digest:
        raise ValueError('Decoded content hash mismatch')
    return parts
