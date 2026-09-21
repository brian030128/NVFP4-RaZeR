from campaign.extension_p75_analysis import DEVICES, MODELS


def test_portability_panel_and_device_order_are_frozen():
    assert MODELS == ("llama8b", "granite8b")
    assert DEVICES == ("a6000", "ada")
