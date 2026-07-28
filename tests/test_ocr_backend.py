import copy
import json
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recognition import LocalOCR, RecognitionEngine, clear_ocr_runtime, ocr_cache_info


class FakeSession:
    def __init__(self, provider):
        self.provider = provider

    def get_providers(self):
        return [self.provider]


class FakeInfer:
    def __init__(self, provider):
        self.session = FakeSession(provider)


class FakeRapidOCR:
    def __init__(self, provider, *, fail_after=0):
        self.text_det = type("TextDet", (), {"infer": FakeInfer(provider)})()
        self.text_cls = type("TextCls", (), {"infer": FakeInfer(provider)})()
        self.text_rec = type("TextRec", (), {"session": FakeInfer(provider)})()
        self.calls = 0
        self.fail_after = fail_after

    def __call__(self, _image):
        self.calls += 1
        if self.fail_after and self.calls > self.fail_after:
            raise RuntimeError("simulated inference failure")
        return [], [0.0, 0.0, 0.0]


def backend_from_kwargs(kwargs):
    if kwargs.get("det_use_cuda"):
        return "cuda", "CUDAExecutionProvider"
    if kwargs.get("det_use_dml"):
        return "dml", "DmlExecutionProvider"
    return "cpu", "CPUExecutionProvider"


class OCRBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.backend_override = os.environ.pop("HUNTER_OCR_BACKEND", None)
        clear_ocr_runtime(engines=True)

    def tearDown(self):
        if self.backend_override is not None:
            os.environ["HUNTER_OCR_BACKEND"] = self.backend_override
        else:
            os.environ.pop("HUNTER_OCR_BACKEND", None)

    def configured(self, backend="auto", fallback=True):
        config = copy.deepcopy(self.config)
        config["recognition_v2"]["ocr_backend"] = backend
        config["recognition_v2"]["ocr_gpu_fallback"] = fallback
        return config

    def test_auto_prefers_cuda_and_reports_actual_session_provider(self):
        calls = []

        def factory(**kwargs):
            backend, provider = backend_from_kwargs(kwargs)
            calls.append(backend)
            return FakeRapidOCR(provider)

        ocr = LocalOCR(
            self.config,
            engine_factory=factory,
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            np_module=np,
        )

        info = ocr.runtime_info()

        self.assertEqual(["cuda"], calls)
        self.assertEqual("cuda", info["active_backend"])
        self.assertEqual("CUDAExecutionProvider", info["actual_provider"])

    def test_auto_falls_back_to_cpu_when_cuda_initialization_fails(self):
        calls = []

        def factory(**kwargs):
            backend, provider = backend_from_kwargs(kwargs)
            calls.append(backend)
            if backend == "cuda":
                raise RuntimeError("CUDA libraries missing")
            return FakeRapidOCR(provider)

        ocr = LocalOCR(
            self.config,
            engine_factory=factory,
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            np_module=np,
        )

        info = ocr.runtime_info()

        self.assertEqual(["cuda", "cpu"], calls)
        self.assertEqual("cpu", info["active_backend"])
        self.assertEqual("CPUExecutionProvider", info["actual_provider"])

    def test_explicit_dml_without_provider_falls_back_only_when_enabled(self):
        calls = []

        def factory(**kwargs):
            backend, provider = backend_from_kwargs(kwargs)
            calls.append(backend)
            return FakeRapidOCR(provider)

        fallback = LocalOCR(
            self.configured("dml", True),
            engine_factory=factory,
            available_providers=["CPUExecutionProvider"],
            np_module=np,
        )
        self.assertEqual("cpu", fallback.runtime_info()["active_backend"])
        self.assertEqual(["cpu"], calls)

        strict = LocalOCR(
            self.configured("dml", False),
            engine_factory=factory,
            available_providers=["CPUExecutionProvider"],
            np_module=np,
        )
        with self.assertRaises(RuntimeError):
            strict.runtime_info()

    def test_gpu_inference_failure_reinitializes_cpu_and_retries(self):
        calls = []

        def factory(**kwargs):
            backend, provider = backend_from_kwargs(kwargs)
            calls.append(backend)
            return FakeRapidOCR(provider, fail_after=1 if backend == "cuda" else 0)

        ocr = LocalOCR(
            self.config,
            engine_factory=factory,
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            np_module=np,
        )
        ocr.runtime_info()

        self.assertEqual([], ocr.read(np.zeros((32, 64, 3), dtype=np.uint8)))
        self.assertEqual(["cuda", "cpu"], calls)
        self.assertEqual("cpu", ocr.active_backend)

    def test_process_cache_reuses_same_frame_roi_and_configuration(self):
        engine = FakeRapidOCR("CPUExecutionProvider")
        ocr = LocalOCR(self.configured("cpu"), engine=engine, np_module=np)
        recognition = RecognitionEngine(self.config, cv2, np, ocr=ocr)
        recognition._current_image_hash = "same-normalized-frame"
        image = np.zeros((80, 120, 3), dtype=np.uint8)

        recognition._read_region(image, (10, 10, 70, 50))
        recognition._read_region(image, (10, 10, 70, 50))

        self.assertEqual(1, engine.calls)
        self.assertEqual({"entries": 1, "hits": 1, "misses": 1}, ocr_cache_info())


if __name__ == "__main__":
    unittest.main()
