import torch

from campaign.extension_heldout_maps import aggregate_scores, masks_for_rule, selector_key


def _state(values):
    # Zero SE makes U_k equal the supplied CE/KL mean.
    v = torch.tensor(values, dtype=torch.float64)
    return {"n": 2, "ce_sum": 2 * v, "ce_sq": 2 * v.square(),
            "kl_sum": 2 * v, "kl_sq": 2 * v.square(), "cross": 2 * v.square()}


def test_draw_mean_recomputes_continuous_scores_not_masks():
    blobs = [{"n16": {"m": _state(v)}} for v in ([-2.0, 1.0], [-1.0, 2.0],
                                                    [-3.0, 3.0], [-4.0, 4.0], [-5.0, 5.0])]
    score, consensus = aggregate_scores(blobs, ["m"], {"m": (16, 128)}, 2.0, "draw_mean")
    assert consensus is None
    torch.testing.assert_close(score["m"].reshape(-1), torch.tensor([-3.0, 3.0], dtype=torch.float64))


def test_fixed_budget_matches_seed0_n16_count():
    blobs = [{"n16": {"m": _state([-2.0, 1.0])}} for _ in range(5)]
    p41 = {"aggregation": "draw_mean", "form": "fixed", "global_k": 2.0}
    _, masks, form, _ = masks_for_rule(blobs, ["m"], {"m": (16, 128)}, p41, 1)
    assert form == "fixed" and int(masks["m"].sum()) == 1


def test_selector_key_respects_old_fallback():
    assert selector_key({"selected_policy": "p50_old_first_order_ce_kl",
                         "fallback_to_old": True}) is None
    assert selector_key({"selected_policy": "p50_layer_output_reconstruction",
                         "fallback_to_old": False}) == "layer_output_reconstruction"
