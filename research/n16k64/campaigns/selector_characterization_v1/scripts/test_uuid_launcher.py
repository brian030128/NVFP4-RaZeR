import unittest
from pathlib import Path
from gpu_run_on_uuid import transformed, BASE_SHA256

class UUIDRestriction(unittest.TestCase):
    def test_only_selection_and_provenance_change(self):
        uuid='GPU-9cec7336-5b30-3f86-35e7-06919156e7da'
        text=transformed(uuid)
        # Reversing the two allowed changes must recover the entire launcher,
        # including its child, phase, ownership, lease and postflight behavior.
        text=text.replace("if g['uuid'] != "+repr(uuid)+" or g['uuid'] in active", "if g['uuid'] in active",1)
        text=text.replace("    rec.update(launcher_base_sha256="+repr(BASE_SHA256)+",allowed_uuid="+repr(uuid)+")\n",'',1)
        self.assertEqual(text,Path(__file__).with_name('gpu_run.py').read_text())

    def test_reject_non_uuid_input(self):
        for value in ['', '0', 'GPU-;command', "GPU-'invalid"]:
            with self.assertRaises(ValueError):transformed(value)

if __name__=='__main__':unittest.main()
