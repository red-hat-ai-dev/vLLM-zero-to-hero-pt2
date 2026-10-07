import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "quantize", Path(__file__).resolve().parents[1] / "quantize.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def checkpoint(**kwargs):
    p = kwargs["save_directory"]
    (p / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed",
                }
            }
        )
    )
    (p / "model.safetensors").write_bytes(b"synthetic test bytes; not real weights")
    (p / "tokenizer_config.json").write_text("{}")


class QuantizeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name) / "result"

    def convert(self, converter):
        with patch.dict(
            sys.modules, llmcompressor=types.SimpleNamespace(model_free_ptq=converter)
        ):
            return module.quantize_checkpoint("test/model", self.output, "cpu")

    def test_completed_conversion_is_published(self):
        self.assertEqual(self.convert(checkpoint)["files"], 1)
        self.assertTrue((self.output / "config.json").exists())

    def test_existing_result_survives_rerun(self):
        self.convert(checkpoint)
        before = (self.output / "model.safetensors").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.convert(checkpoint)
        self.assertEqual((self.output / "model.safetensors").read_bytes(), before)

    def test_failed_conversion_never_publishes(self):
        def broken(**kw):
            checkpoint(**kw)
            raise RuntimeError("download interrupted")

        with self.assertRaises(RuntimeError):
            self.convert(broken)
        self.assertFalse(self.output.exists())

    def test_missing_shard_is_rejected(self):
        def broken(**kw):
            checkpoint(**kw)
            (kw["save_directory"] / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "absent.safetensors"}})
            )

        with self.assertRaisesRegex(ValueError, "shards"):
            self.convert(broken)
        self.assertFalse(self.output.exists())

    def test_missing_metadata_is_rejected(self):
        def broken(**kw):
            checkpoint(**kw)
            (kw["save_directory"] / "config.json").write_text("{}")

        with self.assertRaisesRegex(ValueError, "metadata"):
            self.convert(broken)


class SetupRecoveryTests(unittest.TestCase):
    def test_existing_incomplete_venv_retries_dependency_install(self):
        import os
        import shutil
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copy(Path(__file__).resolve().parents[1] / "quantize.sh", root)
            python = root / "python chosen"
            python.write_text("#!/bin/sh\necho 3.13\n")
            python.chmod(0o755)
            venv = root / ".venv-pt2-quantizer-3.13"
            (venv / "bin").mkdir(parents=True)
            executable = venv / "bin/python"
            executable.write_text(
                '#!/bin/sh\ncase "$*" in *"import llmcompressor"*) exit 1 ;; *"-m pip install"*) echo "$*" >> "$INSTALL_LOG"; exit 7 ;; esac\nexit 0\n'
            )
            executable.chmod(0o755)
            env = dict(
                os.environ,
                PYTHON=str(python),
                QUANTIZER_BACKEND="local",
                HF_HOME=str(root / "cache"),
                INSTALL_LOG=str(root / "install.log"),
            )
            run = subprocess.run(
                ["/bin/sh", str(root / "quantize.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("-m pip install", (root / "install.log").read_text())
            self.assertTrue(executable.exists())


if __name__ == "__main__":
    unittest.main()
