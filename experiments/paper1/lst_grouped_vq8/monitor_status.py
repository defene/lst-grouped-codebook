"""Read live Windows progress files while allowing their writers to rename them."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
from .prepare import OUTPUT


def read_shared(path, tail=None):
    if os.name == 'nt':
        import msvcrt
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 7, None, 3, 128, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        stream = os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), 'rb')
    else:
        stream = Path(path).open('rb')
    with stream:
        if tail is not None:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-tail))
        return stream.read().decode('utf-8', errors='replace')


def main():
    result = {'suite': json.loads(read_shared(OUTPUT/'status.json')), 'models': []}
    for p in sorted((OUTPUT/'full').glob('*/status.json')):
        s = json.loads(read_shared(p))
        result['models'].append({'model': p.parent.name, **{k: s.get(k) for k in
            ('status', 'stage', 'step', 'records_presented', 'validated_epoch', 'utc', 'error')}})
    result['supervisor_error_tail'] = read_shared(OUTPUT/'supervisor.stderr.log', 4000)
    for line in reversed(read_shared(OUTPUT/'gpu.jsonl', 4000).splitlines()):
        try:
            result['gpu'] = json.loads(line)
            break
        except json.JSONDecodeError:
            pass
    for p in sorted((OUTPUT/'full').glob('*.log')):
        result[p.name] = read_shared(p, 1000).splitlines()[-2:]
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
