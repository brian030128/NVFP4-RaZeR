def test_global_k_selection_uses_tolerance_then_worst_then_budget_then_larger_k():
    from campaign.extension_p40_analysis import select_k

    rows = [
        {"k": 0.5, "eligible": False, "median_six_cells": -2.0, "worst_cell": -2.0,
         "total_selected_weights": 1},
        {"k": 1.0, "eligible": True, "median_six_cells": -0.00100, "worst_cell": 0.0002,
         "total_selected_weights": 100},
        {"k": 1.5, "eligible": True, "median_six_cells": -0.00095, "worst_cell": 0.0001,
         "total_selected_weights": 200},
        {"k": 2.0, "eligible": True, "median_six_cells": 0.0, "worst_cell": 0.0,
         "total_selected_weights": 10},
    ]
    assert select_k(rows)["k"] == 1.5
    rows[1]["worst_cell"] = rows[2]["worst_cell"]
    assert select_k(rows)["k"] == 1.0
    rows[1]["total_selected_weights"] = rows[2]["total_selected_weights"]
    assert select_k(rows)["k"] == 1.5
