# Official split loader verification

- Split loader commit: `52e5c41e82d235a4892851e695d0be6f28ce022e`.
- Remote environment: Python 3.10.8, PyTorch 2.1.2+cu118, pytest 8.4.2; this is below the repository's declared PyTorch 2.7 minimum.
- Regression command: `python -m pytest tests/test_pipeline.py::test_mist_archive_preserves_official_train_and_test_cohorts -q`.
- Regression result: **1 passed** in 2.78 s.
- Full suite command: `python -m pytest -q`.
- Full suite result: **20 passed, 1 warning** in 22.06 s. The warning is PyTorch's TypedStorage deprecation from the CLI pipeline test.
- The regression fixture deliberately uses the same source prefix in both original cohorts and confirms all original `valA/valB` pairs remain test while all `trainA/trainB` pairs remain train. It also checks deterministic assignment.
- Full-manifest cache preparation and index audit are pending. Formal training has not started on this corrected split.
