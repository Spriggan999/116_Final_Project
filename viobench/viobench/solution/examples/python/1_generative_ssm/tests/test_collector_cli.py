"""Check collector startup without importing the GPU simulator."""
from pathlib import Path
import subprocess
import sys
import unittest


class CollectorCLI(unittest.TestCase):
    def test_help_before_simulator_imports(self):
        script = Path(__file__).resolve().parents[1] / "collect.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--out", result.stdout)
        self.assertIn("--scenes", result.stdout)
        self.assertIn("--scene-catalog", result.stdout)
        self.assertIn("--compression-level", result.stdout)

    def test_catalog_bounds_checked_before_simulator_imports(self):
        script = Path(__file__).resolve().parents[1] / "collect.py"
        result = subprocess.run(
            [sys.executable, str(script), "--scene-catalog", "hssd", "--scenes", "126"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("125 available hssd scenes", result.stderr)


if __name__ == "__main__":
    unittest.main()
