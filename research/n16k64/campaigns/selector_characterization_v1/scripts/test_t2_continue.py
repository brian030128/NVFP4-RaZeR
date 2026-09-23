import unittest
from t2_continue import choose

class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(model='llama8b',draw='draw1',status='pending_evaluation'),
                   dict(model='mistral7b',draw='draw2',status='pending_evaluation')]

    def test_fixed_order_not_nll(self):
        self.rows[0]['nll']=100
        self.rows[1]['nll']=0
        self.assertEqual(choose(self.rows,[],set()),('llama8b','draw1'))

    def test_one_inflight_per_model(self):
        self.assertEqual(choose(self.rows,[dict(model='llama8b')],set()),('mistral7b','draw2'))

    def test_no_automatic_retry(self):
        self.assertIsNone(choose(self.rows,[],{('llama8b','draw1'),('mistral7b','draw2')}))

if __name__=='__main__':unittest.main()
