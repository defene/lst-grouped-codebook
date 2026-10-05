"""Reversible fixed-bit token container with explicit mask/header overhead."""
import hashlib
import json
import math
import struct
import numpy as np


def pack(indices, vocabulary, input_mask, metadata):
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.min() < 0 or indices.max() >= vocabulary:
        raise ValueError('Invalid token grid')
    width = math.ceil(math.log2(vocabulary))
    bits = (indices.ravel()[:, None].astype(np.uint64) >> np.arange(width, dtype=np.uint64)) & 1
    payload = np.packbits(bits.astype(np.uint8).ravel(), bitorder='little').tobytes()
    mask = np.asarray(input_mask, dtype=bool)
    mask_bytes = np.packbits(mask.ravel(), bitorder='little').tobytes()
    header = dict(metadata, shape=list(indices.shape), vocabulary=vocabulary, bits_per_token=width,
                  mask_shape=list(mask.shape), token_bytes=len(payload), mask_bytes=len(mask_bytes),
                  model_weights_included=False, codec='fixed_bit_v1')
    encoded_header = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    data = b'CVDT1'+struct.pack('<I', len(encoded_header))+encoded_header+payload+mask_bytes
    return data+hashlib.sha256(data).digest()


def unpack(data):
    if data[:5] != b'CVDT1' or len(data) < 41 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError('Corrupt token container')
    length = struct.unpack('<I', data[5:9])[0]
    header = json.loads(data[9:9+length])
    begin = 9+length
    width = header['bits_per_token']
    count = math.prod(header['shape'])
    token_bytes = header['token_bytes']
    if begin+token_bytes+header['mask_bytes']+32 != len(data):
        raise ValueError('Token container length mismatch')
    bits = np.unpackbits(np.frombuffer(data[begin:begin+token_bytes], np.uint8), bitorder='little')[:count*width]
    ids = (bits.reshape(count, width).astype(np.uint64)*(1 << np.arange(width, dtype=np.uint64))).sum(1)
    if ids.max() >= header['vocabulary']:
        raise ValueError('Invalid token ID')
    mask = np.unpackbits(np.frombuffer(data[begin+token_bytes:-32], np.uint8), bitorder='little')[:math.prod(header['mask_shape'])]
    return ids.reshape(header['shape']).astype(np.int64), mask.reshape(header['mask_shape']).astype(bool), header
