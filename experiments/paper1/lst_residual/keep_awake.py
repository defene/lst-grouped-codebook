"""Prevent automatic system sleep only while the named supervisor is alive."""
import argparse
import ctypes
from ctypes import wintypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    handle = kernel.OpenProcess(0x00100000, False, args.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.SetThreadExecutionState(0x80000001):
            raise ctypes.WinError(ctypes.get_last_error())
        print(f'Automatic system sleep prevented while supervisor {args.pid} is alive; display may turn off.', flush=True)
        result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        kernel.CloseHandle(handle)
        print('Supervisor exited; temporary sleep request released.', flush=True)


if __name__ == '__main__':
    main()
