"""vision_client: reading pasted vision results (including JSON fragments)."""

import numpy as np
import pytest

from piece_picking.vision_client import parse_vision_candidates

ONE = """    {
      "object_id": 3,
      "rank": 3,
      "position_m": [
        0.011000001803040504,
        -0.1457500010728836,
        1.366000056267847
      ],
      "normal": [
        0.006712059490382671,
        -0.02153539098799228,
        0.999745477714539
      ],
      "uv": [1033, 678],
      "clearance_valid": false
    },
"""


def test_one_candidate_with_trailing_comma_is_rounded():
    (r,) = parse_vision_candidates(ONE)
    assert r.position_m.tolist() == [0.011, -0.14575, 1.366]                # cut to 0.01 mm
    assert r.normal.tolist() == [0.006712, -0.021535, 0.999745]            # cut to 1e-6
    assert r.rank == 3 and r.info == "object_id 3, rank 3" and r.frame is None


def test_whole_file_gives_all_candidates_in_order():
    text = '{"image": "x.png", "candidates": [' + ONE + ONE.replace('"rank": 3', '"rank": 1') + "]}"
    rs = parse_vision_candidates(text)
    assert [r.rank for r in rs] == [3, 1]


def test_only_key_lines_and_cut_fragments():
    lines = '"position_m": [0.1234567, 0.2, 1.3],\n"normal": [0, 0, -1],'
    (r,) = parse_vision_candidates(lines)
    assert r.position_m.tolist() == [0.12346, 0.2, 1.3] and r.normal.tolist() == [0, 0, -1]
    cut = 'ty": 0.06, "position_m": [0.1, 0.2, 1.3], "normal": [0.0, 0.1, 0.99], "uv": [1033,'   # not JSON
    (r,) = parse_vision_candidates(cut)
    assert r.position_m.tolist() == [0.1, 0.2, 1.3] and r.normal.tolist() == [0.0, 0.1, 0.99]


def test_missing_normal_and_nothing_found():
    (r,) = parse_vision_candidates('{"position_m": [0.1, 0.2, 1.3]}')
    assert r.normal is None
    assert parse_vision_candidates("hello") == []
    assert parse_vision_candidates("") == []


@pytest.mark.parametrize("text", ['{"position_m": [11.0, -145.7, 1366.0]}',     # entered in mm
                                  '{"position_m": [0.1, 0.2, -1.3]}',            # behind the camera
                                  '{"position_m": [0.1, 0.2]}'])
def test_bad_values_are_refused(text):
    with pytest.raises(ValueError):
        parse_vision_candidates(text)
