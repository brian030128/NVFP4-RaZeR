"""Install evaluation policies on a loaded model (weights + activation hooks), with map verification."""
import ctypes
import hashlib
import json
from pathlib import Path

import torch

from campaign import mapio as MIO
from campaign import models as MOD
from campaign import quant as Q
from campaign import tiles as T

# kind -> (weight candidate, activation quantizer or None)
KINDS = {
    'bf16': (None, None),
    'nvfp4': ('nvfp4', 'nvfp4_rows'),
    'four_over_six': ('four_over_six', 'four_over_six_rows'),
    'all_e0m3': ('e0m3', 'four_over_six_rows'),
    'map': ('map', 'four_over_six_rows'),
    'razer_wonly_shared_act': ('razer_e3m3', 'four_over_six_rows'),
    'razer_native_rows': ('razer_e3m3', 'razer_e4m3_rows'),
    'nover6_wonly_shared_act': ('nover6', 'four_over_six_rows'),
    'nover6_native_rows': ('nover6', 'nover6_rows'),
    # historical (non-causal, tensor-wide activation factor) conventions
    'hist_four_over_six': ('four_over_six', 'four_over_six_tensor'),
    'hist_map': ('map', 'four_over_six_tensor'),
    'hist_nvfp4': ('nvfp4', 'nvfp4_tensor'),
}


def latest_complete_run(campaign_root, job_id):
    cands = []
    for d in Path(campaign_root, 'runs').glob(f'{job_id}_attempt*'):
        lr = d / 'launch_record.json'
        if lr.exists() and json.loads(lr.read_text()).get('status') == 'complete':
            cands.append((int(d.name.rsplit('attempt', 1)[1]), d))
    if not cands:
        raise FileNotFoundError(f'no complete run for job {job_id}')
    return sorted(cands)[-1][1]


def resolve_plan(plan, campaign_root):
    """Resolve {"from_job": JOB, "map_policy": P} entries to the recorded map path and SHA-256 of the latest complete attempt."""
    out = []
    for e in plan:
        e = dict(e)
        if e.get('kind') in ('map', 'hist_map') and 'from_job' in e:
            run = latest_complete_run(campaign_root, e['from_job'])
            manifest = run / 'calibration' / 'map_manifest.json'
            if not manifest.exists():
                manifest = run / 'derived_maps' / 'map_manifest.json'
            entries = {m['policy']: m for m in json.loads(manifest.read_text())}
            m = entries[e['map_policy']]
            e.update(map_path=m['path'], map_sha256=m['sha256'], type_block=m['type_block'], resolved_from_run=run.name,
                     expected_total_tiles=m['total_tiles'])
        out.append(e)
    return out


def known_source_manifests(campaign_root):
    """Source-manifest digests of every launched run (the archived campaign source states)."""
    out = set()
    for lr in Path(campaign_root, 'runs').glob('*/launch_record.json'):
        try:
            out.add(json.loads(lr.read_text())['source_manifest_sha256'])
        except Exception:
            pass
    return out


class Installer:
    def __init__(self, key, model, modules, cache='cpu', protocol_id='aligned-primary', campaign_root=None):
        self.key, self.model, self.modules = key, model, modules
        self.spec = MOD.REGISTRY[key]
        self.cache = cache
        self.protocol_id = protocol_id
        self.campaign_root = campaign_root
        self.pristine = {n: m.weight.detach().to('cpu', copy=True) for n, m in modules.items()}
        self.cand = {}
        self.act = None
        self.current = None
        self.shapes = {n: list(m.weight.shape) for n, m in modules.items()}

    def _candidate(self, kind, name):
        cache = self.cand.setdefault(kind, {})
        if name in cache:
            return cache[name].to(self.modules[name].weight.device, non_blocking=True)
        w = self.pristine[name].to(self.modules[name].weight.device)
        q = Q.WEIGHT[kind](w)
        if self.cache == 'cpu' and kind in ('four_over_six', 'e0m3'):
            cache[name] = q.cpu()
        return q

    def verify_map(self, path, sha256, policy_name, type_block, expected_total=None, protocol_id=None):
        """`protocol_id` lets one plan mix arms built under different protocols (e.g. an aligned-primary reference
        against aligned-robustness variants). Each map is still verified against the protocol its plan entry declares."""
        header, masks, digest = MIO.read_map(path, expected_sha256=sha256)
        MIO.verify_for_model(header, model_id=self.spec['model_id'], model_revision=self.spec['revision'],
                             tokenizer_revision=self.spec['revision'],
                             weight_shapes={n: tuple(s) for n, s in self.shapes.items()}, type_block=tuple(type_block),
                             protocol_id=(protocol_id or self.protocol_id), policy_name=policy_name,
                             known_source_manifests=(known_source_manifests(self.campaign_root) if self.campaign_root else None),
                             expected_total_tiles=expected_total)
        return header, masks, digest

    @torch.no_grad()
    def install(self, policy):
        """policy: dict(name, kind, [map_path, map_sha256, map_policy, type_block])."""
        kind = policy['kind']
        wkind, akind = KINDS[kind]
        if self.act is not None:
            self.act.remove()
            self.act = None
        info = dict(name=policy['name'], kind=kind, weight=wkind, activation=akind)
        h = hashlib.sha256()
        selected = 0
        if wkind == 'map':
            tb = tuple(policy['type_block'])
            header, masks, digest = self.verify_map(policy['map_path'], policy['map_sha256'], policy['map_policy'], tb,
                                                    policy.get('expected_total_tiles'), policy.get('protocol_id'))
            info.update(map_path=policy['map_path'], map_sha256=digest, map_reloaded_for_evaluation=True,
                        selected_tiles=header['totals']['selected_tiles'], total_tiles=header['totals']['total_tiles'],
                        type_block=list(tb), map_header_policy=header['policy'], expected_protocol_id=(policy.get('protocol_id') or self.protocol_id))
        for n, m in self.modules.items():
            if wkind is None:
                q = self.pristine[n].to(m.weight.device)
            elif wkind == 'map':
                base = self._candidate('four_over_six', n)
                if masks[n].any():
                    alt = self._candidate('e0m3', n)
                    q = T.apply_mask(base, alt, masks[n], tb)
                else:
                    q = base
            else:
                q = self._candidate(wkind, n)
            m.weight.copy_(q)
            h.update(n.encode())
            h.update(memoryview(q.detach().contiguous().view(torch.uint8).cpu().numpy()))  # zero-copy digest of the installed bytes
            del q
        if akind is not None:
            self.act = Q.ActivationQuant(self.modules, akind, ste=False)
        info['installed_weight_sha256'] = h.hexdigest()
        self.current = info
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)  # return freed host memory to the OS between policies
        except OSError:
            pass
        return info

    def remove(self):
        if self.act is not None:
            self.act.remove()
            self.act = None
