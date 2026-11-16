"""Checks for quota-safe caching and resumable benchmark identity."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import configure_cache,validate_resume


class BenchmarkTests(unittest.TestCase):
    def test_cache_reuses_previous_downloads(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{
            'CONTA_CACHE':'/old/cache','CONTA_SEARCH_PATHS':'/another/cache:/old/cache'}):
            configure_cache(Path(folder)/'conta')
            self.assertEqual(os.environ['CONTA_CACHE'],str(Path(folder)/'conta'))
            self.assertEqual(os.environ['CONTA_SEARCH_PATHS'],'/old/cache:/another/cache')

    def test_resume_accepts_moved_outputs_but_rejects_changed_models(self):
        before=dict(sources=['episode'],smoke_frames=None,binaries={'base':dict(sha256='abc',path='/old/base.wasm')})
        after=dict(sources=['episode'],smoke_frames=None,binaries={'base':dict(sha256='abc',path='/tmp/base.wasm')})
        validate_resume(before,after)
        after['binaries']['base']['sha256']='changed'
        with self.assertRaises(ValueError):validate_resume(before,after)

    def test_resume_rejects_changed_episodes(self):
        before=dict(sources=['episode'],smoke_frames=None,binaries={})
        for after in [dict(sources=['different'],smoke_frames=None,binaries={}),
                      dict(sources=['episode'],smoke_frames=120,binaries={})]:
            with self.assertRaises(ValueError):validate_resume(before,after)

if __name__=='__main__':unittest.main()
