"""Tiny metadata-only tests; no torch, datasets, GPU, or scheduler allocation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('job_monitor',
    Path(__file__).resolve().parents[1] / 'scripts/watch_mixfp4_jobs.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class CompletionNotifications(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        self.records = {'1': dict(state='COMPLETED', exit_code='0:0', name='score'),
                        '2': dict(state='RUNNING', exit_code='0:0', name='validate')}

    @patch.object(monitor.subprocess, 'run')
    def test_restart_deduplicates_but_reports_later_failure(self, run):
        run.return_value.stdout = 'Queued message test'
        monitor.notify_terminal(self.records, self.out, 'same-thread')
        monitor.notify_terminal(self.records, self.out, 'same-thread')
        self.assertEqual(run.call_count, 1)
        self.records['2'].update(state='OUT_OF_MEMORY', exit_code='1:0')
        monitor.notify_terminal(self.records, self.out, 'same-thread')
        self.assertEqual(run.call_count, 2)
        message = run.call_args.args[0][-1]
        self.assertIn('OUT_OF_MEMORY', message)
        self.assertNotIn('"1":', message)
        ledger = json.loads((self.out / 'notifications.json').read_text())
        self.assertEqual(set(ledger['jobs']), {'1', '2'})

    @patch.object(monitor.subprocess, 'run')
    def test_queue_error_is_not_marked_delivered(self, run):
        run.side_effect = subprocess.CalledProcessError(1, ['codex', 'queue'])
        with self.assertRaises(OSError):
            monitor.notify_terminal(self.records, self.out, 'same-thread')
        self.assertFalse((self.out / 'notifications.json').exists())
        run.side_effect = None
        run.return_value.stdout = 'Queued on retry'
        monitor.notify_terminal(self.records, self.out, 'same-thread')
        self.assertEqual(run.call_count, 2)

    @patch.object(monitor.subprocess, 'run')
    def test_no_notification_for_active_jobs(self, run):
        monitor.notify_terminal({'2': self.records['2']}, self.out, 'same-thread')
        run.assert_not_called()

    @patch.object(monitor.subprocess, 'run')
    def test_wrong_thread_ledger_is_rejected(self, run):
        (self.out / 'notifications.json').write_text(json.dumps(dict(thread='other-thread', jobs={})))
        with self.assertRaises(ValueError):
            monitor.notify_terminal(self.records, self.out, 'same-thread')
        run.assert_not_called()

    def test_scheduler_terminal_spellings(self):
        for state in ('COMPLETED', 'CANCELLED by 33785', 'TIMEOUT', 'FAILED+', 'OUT_OF_MEMORY'):
            self.assertTrue(monitor.terminal(dict(state=state)))
        self.assertFalse(monitor.terminal(dict(state='PENDING')))


if __name__ == '__main__':
    unittest.main()
