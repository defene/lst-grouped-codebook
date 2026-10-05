"""Short-lived shared readers and bounded Windows rename retry."""
import json
import time
from eo_data.core import json_write as original_write
from experiments.paper1.lst_v2_5epoch.monitor_status import read_shared


def read(path):
    return json.loads(read_shared(path))


def json_write(path, value):
    for attempt in range(12):
        try:
            return original_write(path, value)
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.05*2**attempt, .5))
