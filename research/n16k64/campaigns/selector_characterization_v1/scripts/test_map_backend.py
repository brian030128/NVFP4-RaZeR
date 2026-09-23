"""Check new maps with the real evaluator reader and tile installer, CPU only."""
import sys
import unittest
import torch
from common import *
sys.path.insert(0,str(PRIMARY/'source/NVFP4-RaZeR-main'))
from campaign import mapio,tiles

class EvaluatorMapIntegration(unittest.TestCase):
    def test_real_reader_accepts_all_frozen_new_maps(self):
        torch.set_num_threads(1)
        rows=load(OUT/'results/MAP_MANIFEST.json')+load(OUT/'results/GRANULARITY_MAP_MANIFEST.json')
        specs=load(OUT/'FROZEN_PROTOCOL.yaml')['models']
        for row in rows:
            header,masks,digest=mapio.read_map(OUT/row['path'],expected_sha256=row['sha256'])
            spec=specs[row['model']]['spec']
            mapio.verify_for_model(header,model_id=spec['model_id'],model_revision=spec['revision'],tokenizer_revision=spec['revision'],
                weight_shapes={x['name']:tuple(x['weight_shape']) for x in header['modules']},type_block=(row['N'],64),
                protocol_id='selector_characterization_v1',policy_name=row['policy'])
            self.assertEqual(maskhash({k:v.numpy() for k,v in masks.items()}),row['payload_sha256'])
    def test_coarse_geometry_does_not_refit_candidates(self):
        base=torch.arange(512*128,dtype=torch.float32).reshape(512,128).bfloat16()
        alt=-base-1
        for N in [8,16,32,64,128,256]:
            mask=torch.zeros((512//N,2),dtype=torch.bool);mask[1,1]=True
            out=tiles.apply_mask(base,alt,mask,(N,64));expected=base.clone();expected[N:2*N,64:128]=alt[N:2*N,64:128]
            self.assertTrue(torch.equal(out,expected));self.assertEqual(out.dtype,base.dtype)
            self.assertTrue(torch.equal(tiles.apply_mask(base,alt,torch.zeros_like(mask),(N,64)),base))
            self.assertTrue(torch.equal(tiles.apply_mask(base,alt,torch.ones_like(mask),(N,64)),alt))

    def test_granularity_plan_install_metadata(self):
        specs=load(OUT/'FROZEN_PROTOCOL.yaml')['models']
        plans=list((OUT/'plans').glob('*_granularity_anchored_v2.json'))+list((OUT/'plans').glob('*_reproduction_only_v1.json'))
        self.assertTrue(plans)
        for path in plans:
            model=path.name.split('_')[0];spec=specs[model]['spec']
            for entry in load(path):
                if entry['kind']!='map':continue
                header,_,_=mapio.read_map(entry['map_path'],expected_sha256=entry['map_sha256'])
                kwargs=dict(model_id=spec['model_id'],model_revision=spec['revision'],tokenizer_revision=spec['revision'],
                    weight_shapes={x['name']:tuple(x['weight_shape']) for x in header['modules']},
                    type_block=entry['type_block'],protocol_id=entry['protocol_id'])
                mapio.verify_for_model(header,policy_name=entry['map_policy'],**kwargs)
                if entry['name'] in ['n8_reproduction','n16_reproduction']:
                    with self.assertRaises(mapio.MapVerificationError):
                        mapio.verify_for_model(header,policy_name=entry['name'],**kwargs)

if __name__=='__main__':unittest.main()
