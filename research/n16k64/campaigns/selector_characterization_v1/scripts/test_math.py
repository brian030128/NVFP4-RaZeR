"""Acceptance checks isolate set, covariance, ownership and paired inference bugs."""
import unittest
import numpy as np
from common import *

class Correctness(unittest.TestCase):
    def test_nested_containment_not_ranking(self):
        a=np.arange(39000)<13000;b=np.arange(39000)<26000;c=np.ones(39000,bool)
        self.assertAlmostEqual(pair(a,c)['jaccard'],1/3);self.assertEqual(pair(a,c)['containment_a_to_b'],1)
        self.assertEqual(pair(a,b)['containment_a_to_b'],1)
        self.assertFalse(np.array_equal(np.argsort(np.arange(13000)),np.argsort(-np.arange(13000))))
    def test_random_expected_intersection(self):
        r=null_pairs([1000,500],[150,200],[350,100],31,1000)
        self.assertLess(abs(r['null_intersection_mean']-92.5),1)
    def test_correlated_parent_moments(self):
        x=np.arange(-64,64,dtype=float);children=np.stack([x,x,-x],1)
        mu,se=mean_se_array(children.sum(1)[:,None]);_,s=mean_se_array(children)
        self.assertAlmostEqual(se[0],s[0]);self.assertNotAlmostEqual(se[0],np.sqrt((s*s).sum()))
        self.assertTrue((se<=s.sum()+1e-12).all())
    def test_regret_example_and_zero(self):
        r=regret(np.array([[-5.,-4,4,3],[0,0,0,0]]),np.array([-2.,0]),np.zeros(2),np.zeros(2,bool))
        self.assertEqual(r['ownership'][0],7);self.assertEqual(r['veto'][0],2);self.assertEqual(r['total'][0],9)
        self.assertTrue(np.isnan(r['cancellation'][1]))
    def test_all_children_pass_and_weighted_estimator(self):
        rng=np.random.default_rng(8);x=rng.normal(-10,.1,(128,4));m,s=mean_se_array(x)
        self.assertTrue((m+3*s<0).all());pm,ps=mean_se_array(x.sum(1)[:,None]);self.assertLess((pm+3*ps)[0],0)
        w=np.arange(1,129);w=w/w.sum();wm=w@x
        # Unequal-weight estimator uses squared weights, not n-token pseudo-replication.
        cov=(x-wm).T@((x-wm)*w[:,None])/(1-(w*w).sum())
        se=np.sqrt(np.diag(cov)*(w*w).sum());sp=np.sqrt(cov.sum()*(w*w).sum())
        self.assertLessEqual(sp,se.sum()+1e-12)
    def test_joint_matched_empty_zero_nan_tie(self):
        ce=np.array([-3.,-2,0,1]);kl=np.array([1.,-4,0,-1]);joint=(ce<0)&(kl<0)
        self.assertEqual(joint.sum(),1);a,t=topmask(ce,int(joint.sum()));self.assertEqual(a.sum(),joint.sum())
        a,t=topmask(np.ones(4),2);self.assertEqual(a.tolist(),[True,True,False,False]);self.assertEqual(t,1)
        self.assertIsNone(pair(np.zeros(4,bool),np.zeros(4,bool))['jaccard'])
        np.testing.assert_array_equal(zscore(np.array([-1.,0,1]),np.zeros(3)),[-np.inf,0,np.inf])
        with self.assertRaises(AssertionError):zscore(np.array([np.nan]),np.ones(1))
    def test_hierarchical_aggregation(self):
        x=np.arange(128*16,dtype=float).reshape(128,16)
        direct=aggregate_scores(x,64,128,32)
        n16=aggregate_scores(x,64,128,16).reshape(128,4,2)
        via=n16.reshape(128,2,2,2).sum(2).reshape(128,-1)
        np.testing.assert_array_equal(direct,via);self.assertEqual(direct.sum(),x.sum())
    def test_token_weighted_and_joint_bootstrap(self):
        loss=np.array([[1.,99.],[2,198.]])
        est,bs=paired_bootstrap(loss,np.array([1.,9.]),2000,17)
        self.assertEqual(est[0],10);np.testing.assert_allclose(bs[:,1],2*bs[:,0]);self.assertNotEqual(est[0],(1+11)/2)
    def test_primary_map_anchor_audit(self):
        p=OUT/'results/INPUTS.json'
        if not p.exists():self.skipTest('T0 anchor audit must run first')
        rows=load(p);self.assertEqual(len(rows),15)
        for r in rows:
            self.assertEqual(len(r['maps']),3)
            for m in r['maps']:self.assertEqual(sha(REPO/m['path']),m['sha256'])

if __name__=='__main__':unittest.main()
