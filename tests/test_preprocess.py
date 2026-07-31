#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openlrc.preprocess import Preprocessor

DATA_DIR = Path(__file__).parent / "data"

# Inject lightweight fakes for the optional noise-suppression stack so these
# tests run without the openlrc[full] extra installed.
_dpdfnet = types.ModuleType("dpdfnet")
_dpdfnet.enhance = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("dpdfnet", _dpdfnet)

_librosa = types.ModuleType("librosa")
_librosa.load = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("librosa", _librosa)


class _FakeSoundFile:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, data):
        pass


_soundfile = types.ModuleType("soundfile")
_soundfile.SoundFile = _FakeSoundFile
sys.modules.setdefault("soundfile", _soundfile)


class TestPreprocessor(unittest.TestCase):
    def tearDown(self) -> None:
        preprocessed_path = DATA_DIR / "preprocessed"
        shutil.rmtree(preprocessed_path, ignore_errors=True)

    @patch.object(_dpdfnet, "enhance")
    @patch.object(_librosa, "load")
    def test_noise_suppression_returns_path_objects(self, mock_load, mock_enhance):
        chunk_size = 180
        mock_sr = 16000
        mock_audio_size = chunk_size * 5

        mock_enhance.return_value = np.zeros(chunk_size * mock_sr)
        mock_load.return_value = (np.zeros(mock_audio_size * mock_sr), mock_sr)

        preprocessor = Preprocessor("audio.wav")
        ns_paths = preprocessor.noise_suppression(preprocessor.audio_paths)
        self.assertIsInstance(ns_paths, list)
        self.assertIsInstance(ns_paths[0], Path)
        self.assertEqual(mock_enhance.call_count, 5)

    @patch.object(_dpdfnet, "enhance")
    @patch.object(_librosa, "load")
    @patch("openlrc.preprocess.Path.unlink")
    def test_noise_suppression_shape_mismatch_removes_partial_file(self, mock_unlink, mock_load, mock_enhance):
        chunk_size = 180
        mock_sr = 16000
        mock_audio_size = chunk_size * 5

        mock_enhance.return_value = np.zeros(chunk_size * mock_sr + 1)
        mock_load.return_value = (np.zeros(mock_audio_size * mock_sr), mock_sr)

        preprocessor = Preprocessor("audio.wav")
        with self.assertRaises(ValueError):
            preprocessor.noise_suppression(preprocessor.audio_paths)
        mock_unlink.assert_called_once()

    @patch("openlrc.preprocess.FFmpegNormalize")
    def test_loudness_normalization_returns_path_objects(self, mock_norm):
        mock_norm.return_value.run_normalization.return_value = None
        preprocessor = Preprocessor(DATA_DIR / "test_audio.wav")
        ln_paths = preprocessor.loudness_normalization(preprocessor.audio_paths)
        self.assertIsInstance(ln_paths, list)
        self.assertIsInstance(ln_paths[0], Path)

    @patch("openlrc.preprocess.Path.rename")
    @patch("openlrc.preprocess.Preprocessor.noise_suppression")
    @patch("openlrc.preprocess.Preprocessor.loudness_normalization")
    def test_run_returns_path_objects(self, mock_loudness_normalization, mock_noise_suppression, mock_rename):
        mock_rename.return_value = Path("audio_processed.wav")
        mock_noise_suppression.return_value = [Path("audio_ns.wav")]
        mock_loudness_normalization.return_value = [Path("audio_ln.wav")]
        preprocessor = Preprocessor("audio.wav")
        final_processed = preprocessor.run()
        self.assertIsInstance(final_processed, list)
        self.assertIsInstance(final_processed[0], Path)

    def test_preprocessor_raises_exception_when_audio_paths_is_not_a_list_or_a_string(self):
        with self.assertRaises(TypeError):
            Preprocessor(123)

    def test_noise_suppression_missing_optional_deps_has_quoted_install_hint(self):
        preprocessor = Preprocessor("audio.wav")
        with patch.dict(sys.modules, {"dpdfnet": None}):
            with self.assertRaisesRegex(ImportError, r"pip install 'openlrc\[full\]'"):
                preprocessor.noise_suppression(preprocessor.audio_paths)
