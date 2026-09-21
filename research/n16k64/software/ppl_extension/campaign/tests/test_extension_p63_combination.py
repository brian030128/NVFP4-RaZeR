from campaign.extension_p63_combination import ARMS, choose, quality_spec


def test_quality_selection_tie_prefers_fewer_operations():
    rows = [{"arm": arm, "safe": True, "median_six_cells": -0.001,
             "worst_cell": 0.0} for arm in ARMS]
    assert choose(rows)["arm"] == "p63_baseline"


def test_unsafe_candidate_is_ineligible():
    rows = [{"arm": "p63_baseline", "safe": True, "median_six_cells": 0.0,
             "worst_cell": 0.0},
            {"arm": "p63_combined", "safe": False, "median_six_cells": -1.0,
             "worst_cell": 0.002}]
    assert choose(rows)["arm"] == "p63_baseline"


def test_quality_spec_is_one_gpu_ada_and_serializable():
    row = quality_spec("llama8b", "P63_gate")
    assert row["gpus"] == 1 and row["gpu_model"] == "ada"
    assert row["depends_on"] == ["P63_gate"]
