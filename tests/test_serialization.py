import json

import numpy as np
import pytest

from spine_pipeline.serialization import to_jsonable


def test_raw_float32_is_not_serializable_baseline():
    # Documents the original crash: a bare numpy float32 cannot be dumped.
    with pytest.raises(TypeError):
        json.dumps(np.float32(1.5))


def test_float32_scalar_becomes_python_float():
    out = to_jsonable(np.float32(33.33))
    assert isinstance(out, float)
    json.dumps(out)  # must not raise


def test_various_numpy_scalars():
    assert isinstance(to_jsonable(np.int64(5)), int)
    assert isinstance(to_jsonable(np.float64(2.0)), float)
    assert isinstance(to_jsonable(np.bool_(True)), bool)


def test_ndarray_becomes_list():
    assert to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]


def test_nested_structure_serializes_without_string_coercion():
    data = {
        "global_metrics": {"k": np.float64(2.0), "n": np.int64(5)},
        "per_vertebra": [
            {"ap_width_mm": np.float32(12.3), "arr": np.array([1, 2, 3])},
        ],
    }
    out = to_jsonable(data)
    restored = json.loads(json.dumps(out))
    assert restored["global_metrics"]["n"] == 5
    assert restored["per_vertebra"][0]["arr"] == [1, 2, 3]
    # numbers stay numbers (not strings, unlike json.dump(default=str))
    assert isinstance(restored["per_vertebra"][0]["ap_width_mm"], float)


def test_plain_types_pass_through():
    payload = {"a": [1, "x", 2.0, None, True]}
    assert to_jsonable(payload) == payload
