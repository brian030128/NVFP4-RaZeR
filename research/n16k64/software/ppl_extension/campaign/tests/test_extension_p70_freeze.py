from campaign.extension_p70_freeze import MODELS


def test_no_swap_heldout_model_order_is_fixed():
    assert MODELS == ("granite8b", "falcon3_10b")
