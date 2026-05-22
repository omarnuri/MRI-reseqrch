"""JSON-safe conversion of NumPy values.

Root-cause fix for the notebook's `TypeError: Object of type float32 is not
JSON serializable`: NumPy scalars (e.g. produced by `round(int * zooms, 2)`,
where `header.get_zooms()` returns float32) and arrays are not JSON
serializable. `to_jsonable` recursively converts them to plain Python types so
`json.dump` succeeds without coercing numbers to strings.
"""

import numpy as np


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
