"""OS-held per-output writer lock; automatically released after a crash."""
from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def build_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory/'.prepare.lock').open('a+b')
    if stream.seek(0,2) == 0:
        stream.write(b'0'); stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        stream.close()
        raise RuntimeError(f'Another preprocessing writer is using {directory}') from e
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == 'nt':
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()
