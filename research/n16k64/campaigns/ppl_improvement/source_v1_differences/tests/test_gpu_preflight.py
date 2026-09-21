import json
import os
import signal

import pytest

from campaign import gpu_preflight as gp

A6000 = 'NVIDIA RTX A6000'
ADA = 'NVIDIA RTX 6000 Ada Generation'
ME = os.getuid()
OTHER = ME + 4242


def gpus(*names):
    return [dict(index=i, uuid=f'GPU-{i:08d}-aaaa-bbbb-cccc-{i:012d}', name=n, memory_total_mib=49140,
                 memory_used_mib=2, utilization_gpu_pct=0, pci_bus_id=f'00000000:{i+1:02X}:00.0')
            for i, n in enumerate(names)]


SEVEN = gpus(A6000, A6000, A6000, A6000, ADA, ADA, ADA)


def make(gpu_list, apps=(), owners=None, nvml=True):
    owners = owners or {}

    def gq():
        return [dict(g) for g in gpu_list]

    def aq():
        return [dict(a) for a in apps]

    def nq():
        state = {g['uuid']: set() for g in gpu_list}
        for a in apps:
            state[a['gpu_uuid']].add(a['pid'])
        return state

    def oq(pid):
        uid, cg = owners.get(pid, (ME, ''))
        return uid, f'user{uid}', 1000, cg

    return dict(gpu_query=gq, app_query=aq, nvml_query=nq if nvml else None, owner_query=oq, uid=ME)


def env_for(*indices, table=SEVEN, uuid=True):
    ids = [table[i]['uuid'] if uuid else str(table[i]['index']) for i in indices]
    return {'CUDA_VISIBLE_DEVICES': ','.join(ids)}


@pytest.fixture(autouse=True)
def forbid_kill(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('preflight must never terminate a process')
    monkeypatch.setattr(os, 'kill', boom)
    monkeypatch.setattr(os, 'killpg', boom)


def test_zero_visible_gpus_fails_for_gpu_run():
    r = gp.check('t', env={'CUDA_VISIBLE_DEVICES': ''}, **make(SEVEN))
    assert not r['passed'] and any('0 visible' in s for s in r['reasons'])


def test_unset_visibility_on_seven_gpu_host_fails():
    r = gp.check('t', env={}, **make(SEVEN))
    assert not r['passed'] and any('exceeds' in s for s in r['reasons'])


def test_four_visible_gpus_fail():
    r = gp.check('t', env=env_for(0, 1, 2, 3), **make(SEVEN))
    assert not r['passed'] and any('exceeds the per-run maximum' in s for s in r['reasons'])


def test_duplicate_identifiers_fail():
    r = gp.check('t', env={'CUDA_VISIBLE_DEVICES': '0,0'}, **make(SEVEN))
    assert not r['passed'] and any('duplicate' in s for s in r['reasons'])


def test_remapped_index_and_uuid_of_same_device_fail():
    r = gp.check('t', env={'CUDA_VISIBLE_DEVICES': f'0,{SEVEN[0]["uuid"]}'}, **make(SEVEN))
    assert not r['passed']
    assert any('duplicate' in s for s in r['reasons']) and any('ambiguous' in s for s in r['reasons'])


def test_nonexistent_or_garbage_identifier_fails():
    assert not gp.check('t', env={'CUDA_VISIBLE_DEVICES': '9'}, **make(SEVEN))['passed']
    assert not gp.check('t', env={'CUDA_VISIBLE_DEVICES': 'MIG-x'}, **make(SEVEN))['passed']


def test_mixed_a6000_ada_fails():
    r = gp.check('t', env=env_for(3, 4), **make(SEVEN))
    assert not r['passed'] and any('mixed GPU models' in s for s in r['reasons'])


def test_disallowed_model_fails():
    table = gpus('NVIDIA H100 80GB HBM')
    r = gp.check('t', env={}, **make(table))
    assert not r['passed'] and any('not permitted' in s for s in r['reasons'])


def test_foreign_pid_fails_without_termination():
    apps = [dict(gpu_uuid=SEVEN[2]['uuid'], pid=555, used_memory_mib=100, process_name='python')]
    r = gp.check('t', env=env_for(2), **make(SEVEN, apps, owners={555: (OTHER, '')}))
    assert not r['passed'] and any('foreign compute process PID 555' in s for s in r['reasons'])
    assert r['compute_processes'][0]['owner_uid'] == OTHER


def test_foreign_pid_on_unselected_gpu_does_not_block():
    apps = [dict(gpu_uuid=SEVEN[0]['uuid'], pid=555, used_memory_mib=100, process_name='python')]
    r = gp.check('t', env=env_for(2), **make(SEVEN, apps, owners={555: (OTHER, '')}))
    assert r['passed'], r['reasons']


def test_same_user_preexisting_process_fails_in_strict_mode():
    apps = [dict(gpu_uuid=SEVEN[2]['uuid'], pid=777, used_memory_mib=100, process_name='python')]
    q = make(SEVEN, apps, owners={777: (ME, '/user.slice/other')})
    assert not gp.check('t', env=env_for(2), **q)['passed']
    ok = gp.check('t', env=env_for(2), own_cgroup='docker-abc', **make(SEVEN, apps, owners={777: (ME, '0::/system.slice/docker-abc.scope')}))
    assert ok['passed'], ok['reasons']


@pytest.mark.parametrize('indices', [(2,), (2, 3), (4, 5, 6)])
def test_clean_one_two_three_gpu_allocations_pass(indices):
    r = gp.check('t', env=env_for(*indices), expected_uuids=[SEVEN[i]['uuid'] for i in indices], **make(SEVEN))
    assert r['passed'], r['reasons']
    assert r['visible']['count'] == len(indices)


def test_allocation_mismatch_fails():
    r = gp.check('t', env=env_for(2), expected_uuids=[SEVEN[3]['uuid']], **make(SEVEN))
    assert not r['passed'] and any('differ from the allocation' in s for s in r['reasons'])


def test_unavailable_query_fails_closed():
    def broken():
        raise gp.QueryError('nvidia-smi: command not found')
    q = make(SEVEN)
    q['gpu_query'] = broken
    r = gp.check('t', env=env_for(2), **q)
    assert not r['passed'] and r['reasons'][0].startswith('fail-closed')


def test_malformed_query_output_fails_closed():
    with pytest.raises(gp.QueryError):
        gp.smi_gpus(run=lambda cmd: 'garbage without commas\n')
    with pytest.raises(gp.QueryError):
        gp.smi_compute_apps(run=lambda cmd: 'GPU-x, notapid, 3, python\n')
    with pytest.raises(gp.QueryError):
        gp.smi_gpus(run=lambda cmd: '')

    def bad_run(cmd):
        raise gp.QueryError('rc=9')
    q = make(SEVEN)
    q['gpu_query'] = lambda: gp.smi_gpus(run=bad_run)
    assert not gp.check('t', env=env_for(2), **q)['passed']


def test_real_subprocess_failure_is_query_error():
    with pytest.raises(gp.QueryError):
        gp._run(['/nonexistent/nvidia-smi-binary'])


def test_nvml_disagreement_fails_closed():
    apps = [dict(gpu_uuid=SEVEN[2]['uuid'], pid=555, used_memory_mib=100, process_name='python')]
    q = make(SEVEN, apps, owners={555: (ME, 'docker-abc')})
    q['nvml_query'] = lambda: {g['uuid']: set() for g in SEVEN}  # hides the PID
    r = gp.check('t', env=env_for(2), own_cgroup='docker-abc', **q)
    assert not r['passed'] and any('disagree' in s for s in r['reasons'])


def test_unresolvable_owner_fails_closed():
    apps = [dict(gpu_uuid=SEVEN[2]['uuid'], pid=999, used_memory_mib=100, process_name='?')]
    q = make(SEVEN, apps)

    def no_owner(pid):
        raise gp.QueryError('cannot resolve owner of PID 999')
    q['owner_query'] = no_owner
    r = gp.check('t', env=env_for(2), **q)
    assert not r['passed'] and any('cannot resolve owner' in s for s in r['reasons'])


def test_smi_parsers_accept_real_formats():
    g = gp.smi_gpus(run=lambda cmd: '0, GPU-DEVICE-00, NVIDIA RTX A6000, 49140, 2, 0, 00000000:01:00.0\n')
    assert g[0]['name'] == A6000 and g[0]['index'] == 0
    assert gp.smi_compute_apps(run=lambda cmd: '') == []
    a = gp.smi_compute_apps(run=lambda cmd: 'GPU-DEVICE-00, 52917, 21530, .conda/bin/python\n')
    assert a[0]['pid'] == 52917 and a[0]['used_memory_mib'] == 21530


def test_json_round_trip_and_schema(tmp_path):
    rec = gp.check('before', env=env_for(2), expected_uuids=[SEVEN[2]['uuid']], allocation_id='lease-1', **make(SEVEN))
    gp.validate(rec)
    path, digest = gp.write_record(rec, tmp_path / 'gpu_preflight_before.json')
    loaded = json.loads(open(path).read())
    assert loaded == json.loads(json.dumps(rec)) and gp.validate(loaded)
    assert oct(os.stat(path).st_mode & 0o777) == oct(0o444)
    with pytest.raises(FileExistsError):
        gp.write_record(rec, tmp_path / 'gpu_preflight_before.json')
    bad = dict(loaded, passed=True, reasons=['x'])
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        gp.validate(bad)


def test_cpu_only_run_needs_no_gpu():
    q = make(SEVEN)
    q['gpu_query'] = lambda: (_ for _ in ()).throw(gp.QueryError('no gpu'))
    r = gp.check('cpu', gpu_run=False, env={'CUDA_VISIBLE_DEVICES': ''}, **q)
    assert r['passed']


def test_host_after_process_classification_is_fail_closed():
    from campaign import launcher as LA
    inspect = '[{"State": {"FinishedAt": "2026-09-11T19:17:14.860328489Z"}}]'
    fin = LA.container_finished_epoch(inspect)
    assert abs(fin - (1789154234 + 0.860328489)) < 1e-6
    assert LA.container_finished_epoch('not json') is None
    clk = 100
    check_wall = fin + 10.0            # host after-check 10 s after container exit
    uptime = 5000.0                    # seconds since boot at the check
    boot_wall = check_wall - uptime
    def proc(pid, started_wall):
        return dict(pid=pid, owner='other', start_ticks=int(round((started_wall - boot_wall) * clk)))
    procs = [proc(1, fin + 5.0), proc(2, fin + 0.5), proc(3, fin - 30.0), dict(pid=4, owner='other', start_ticks=None)]
    over, post = LA.classify_after_processes(procs, fin, check_wall, uptime, clk=clk, margin=1.0)
    assert [p['pid'] for p in post] == [1]
    assert sorted(p['pid'] for p in over) == [2, 3, 4]      # within margin, before exit, and unknown start all overlap
    over2, post2 = LA.classify_after_processes(procs[:1], None, check_wall, uptime, clk=clk)
    assert post2 == [] and len(over2) == 1                   # unknown container exit time -> fail closed


def test_per_gpu_quiet_window_counts_only_other_users(tmp_path):
    import getpass
    import json as _json
    from campaign import launcher as LA
    ev = tmp_path / 'allocator' / 'events'
    ev.mkdir(parents=True)
    a, b, c = 'GPU-DEVICE-00', 'GPU-DEVICE-01', 'GPU-DEVICE-02'
    (ev / 'x.json').write_text(_json.dumps(dict(reason=f"foreign/non-job compute process on leased GPU: [{{'gpu_uuid': '{a}', 'pid': 1, 'owner': '<REDACTED_USER>'}}]")))
    (ev / 'y.json').write_text(_json.dumps(dict(reason=f"foreign/non-job compute process on leased GPU: [{{'gpu_uuid': '{b}', 'pid': 2, 'owner': '<REDACTED_USER>'}}]")))
    cot = LA.cotenancy_gpus(tmp_path)
    assert cot == {a}
    assert LA.quiet_seconds(a, cot) == LA.QUIET_SECONDS and LA.quiet_seconds(b, cot) == LA.SHORT_QUIET_SECONDS == LA.quiet_seconds(c, cot)


def test_phase_gate_requests_a_long_retry_envelope(tmp_path, monkeypatch):
    """A phase-gate trip discards all in-flight work, so the in-job gate must retry the device query for far longer
    than the host-side default before failing closed."""
    import inspect
    # read the real signature before patching: the host-side default stays short so launcher checks and tests are fast
    params = inspect.signature(gp.check).parameters
    assert params['retries'].default == 3 and params['retry_sleep'].default == 0.25
    captured = {}

    def fake_check(phase, **kw):
        captured.update(kw)
        captured['phase'] = phase
        raise RuntimeError('captured')

    monkeypatch.setattr(gp, 'check', fake_check)
    gate = gp.PhaseGate(tmp_path, gpu_run=True)
    with pytest.raises(RuntimeError):
        gate('load_model')
    assert captured['phase'] == 'load_model'
    assert captured['retries'] >= 5, captured
    assert captured['retry_sleep'] >= 0.25, captured
