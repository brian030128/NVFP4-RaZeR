import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from gpu_run_wait import inventory_or_wait, transformed, PARENT_SHA
from gpu_run_on_uuid import transformed as parent_transform

class InventoryWaitTests(unittest.TestCase):
    def test_transformation_leaves_every_other_check_intact(self):
        uuid='GPU-9cec7336-5b30-3f86-35e7-06919156e7da'
        source=transformed(uuid).replace('inventory_or_wait(gp,root)','gp.smi_gpus()',1)
        source=source.replace("    rec.update(wait_adapter_parent_sha256="+repr(PARENT_SHA)+")\n",'',1)
        self.assertEqual(source,parent_transform(uuid))

    def test_query_failure_is_logged_not_accepted(self):
        class QueryError(Exception):pass
        def query():raise QueryError('fixture timeout')
        gp=SimpleNamespace(QueryError=QueryError,smi_gpus=query)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.assertEqual(inventory_or_wait(gp,root),[])
            row=json.loads((root/'inventory_query_errors.jsonl').read_text())
            self.assertFalse(row['gpu_launched']);self.assertEqual(row['status'],'unavailable_before_launch')

    def test_success_preserves_inventory_and_programming_errors_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gpus=[dict(uuid='fixture')]
            gp=SimpleNamespace(QueryError=ValueError,smi_gpus=lambda:gpus)
            self.assertIs(inventory_or_wait(gp,root),gpus)
            self.assertFalse(list(root.iterdir()))
            def broken():raise TypeError('not an availability failure')
            gp.smi_gpus=broken
            with self.assertRaises(TypeError):inventory_or_wait(gp,root)

if __name__=='__main__':unittest.main()
