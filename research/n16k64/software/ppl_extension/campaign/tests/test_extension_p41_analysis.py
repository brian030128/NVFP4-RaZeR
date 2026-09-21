def test_p41_selection_uses_eligibility_then_median_tolerance_worst_jaccard_weights():
    from campaign.extension_p41_analysis import select_candidate

    rows = [
        {"aggregation": "bad", "form": "natural", "eligible": False, "median_six_cells": -9,
         "worst_cell": -9, "median_lodo_jaccard": 1, "total_selected_weights": 1},
        {"aggregation": "a", "form": "natural", "eligible": True, "median_six_cells": -0.001,
         "worst_cell": 0.0002, "median_lodo_jaccard": 0.9, "total_selected_weights": 100},
        {"aggregation": "b", "form": "fixed", "eligible": True, "median_six_cells": -0.00095,
         "worst_cell": 0.0001, "median_lodo_jaccard": 0.8, "total_selected_weights": 50},
    ]
    assert select_candidate(rows)["aggregation"] == "b"
    rows[1]["worst_cell"] = rows[2]["worst_cell"]
    assert select_candidate(rows)["aggregation"] == "a"
    rows[1]["median_lodo_jaccard"] = rows[2]["median_lodo_jaccard"]
    assert select_candidate(rows)["aggregation"] == "b"
