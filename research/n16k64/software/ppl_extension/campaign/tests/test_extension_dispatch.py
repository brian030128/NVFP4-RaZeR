def test_dispatch_exposes_only_frozen_dynamic_action(monkeypatch):
    import sys
    import campaign.extension_dispatch as module

    called = []
    monkeypatch.setattr(module.runtime, "sha256_file", lambda path: module.PROTOCOL)
    monkeypatch.setattr(module, "load", lambda path: {"global_k": 1.5})
    monkeypatch.setattr(module, "p41", lambda model, k: called.append((model, k)))
    monkeypatch.setattr(sys, "argv", ["dispatch", "p41-map", "--model", "qwen4b"])
    module.main()
    assert called == [("qwen4b", 1.5)]
