# OCR backends

`recognition_v2.ocr_backend` accepts `auto`, `cuda`, `dml`, or `cpu`.
`auto` probes CUDA, then DirectML, then CPU. Every selected backend must initialize
all three RapidOCR sessions with the expected Provider and pass a minimal inference
self-test. GPU initialization or inference failures fall back to CPU when
`ocr_gpu_fallback` is enabled.

The ONNX Runtime distributions conflict. Install exactly one in an environment:

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
python -m pip install -r requirements-ocr-cuda.txt
```

Use `requirements-ocr-dml.txt` instead for DirectML. RapidOCR itself must already be
installed from `requirements.txt`; do not reinstall the base requirements after
replacing ONNX Runtime without checking the resulting Provider list.

Verify the runtime rather than the package name:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python benchmark_ocr.py --backend cpu --backend auto --iterations 5 --replay
```

CUDA also requires a compatible CUDA/cuDNN runtime. A visible GPU in `nvidia-smi`
does not by itself prove that `CUDAExecutionProvider` can initialize.

Use `python run_tests.py` for the fast suite. It forces CPU and excludes full real
OCR replay. Use `python run_tests.py --profile full` for milestone acceptance.
