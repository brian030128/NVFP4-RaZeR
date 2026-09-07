"""Exact-placement optimization: keep resident parameters, offload activations."""
import contextlib
import time
import torch
import analyze_task_sensitivity as task


@contextlib.contextmanager
def offload_intermediates(model):
    # Saving resident frozen parameters to CPU and loading them back wastes two
    # whole-model PCIe transfers each backward. Keep their existing storage;
    # native autograd views (including transposes) share that storage pointer.
    storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
    def pack(t):
        if t.device.type == 'cpu' or t.untyped_storage().data_ptr() in storage:
            return t.device, t.detach()
        cpu = torch.empty_like(t, device='cpu', pin_memory=True)
        cpu.copy_(t.detach())
        return t.device, cpu
    def unpack(saved):
        device, t = saved
        return t.to(device, non_blocking=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def fast_scores(model, modules, base, alt, batches):
    # During scoring, the live weights are exactly the baseline. Keep alternate
    # BF16 candidates resident where memory permits. Subtraction remains FP32;
    # storing a rounded BF16 delta here would change the algorithm.
    gpu_base = {n: m.weight.detach() for n, m in modules.items()}
    required = {}
    for n, m in modules.items():
        d = m.weight.device
        required[d] = required.get(d, 0)+alt[n].numel()*alt[n].element_size()
    can_cache = all(torch.cuda.mem_get_info(d)[0] > size+12*2**30 for d, size in required.items())
    gpu_alt = {n: alt[n].to(m.weight.device) for n, m in modules.items()} if can_cache else alt
    print(f'SCORE BACKEND resident_parameters=True resident_alternates={can_cache}', flush=True)
    try:
        with offload_intermediates(model):
            return task.score_tiles(model, modules, gpu_base, gpu_alt, batches)
    finally:
        del gpu_alt, gpu_base


_verified = set()
def checked_scores(model, modules, base, alt, batches, use_optimized=False):
    # The target rejected optimized storage on an exact score comparison.
    # Scientific runs use the original backend; keep the opt-in diagnostic
    # available to reproduce that rejection, never enable it silently.
    if not use_optimized:
        print('SCORE BACKEND original save_on_cpu; optimized path disabled', flush=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return task.score_tiles(model, modules, base, alt, batches)
    if id(model) not in _verified:
        # Exercise every module and every score on two real windows before using
        # the faster placement. No tolerance: forward values and moments match.
        start = time.time()
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            reference = task.score_tiles(model, modules, base, alt, batches[:2])
        old_seconds = time.time()-start
        start = time.time()
        actual = fast_scores(model, modules, base, alt, batches[:2])
        fast_seconds = time.time()-start
        assert reference[2] == actual[2]
        for r, a in zip(reference[:2], actual[:2]):
            assert r.keys() == a.keys()
            for n in r:
                assert torch.equal(r[n], a[n]), n
        _verified.add(id(model))
        print(f'PASS exact score backend: reference={old_seconds:.2f}s optimized={fast_seconds:.2f}s', flush=True)
    return fast_scores(model, modules, base, alt, batches)
