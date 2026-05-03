import numpy as np

from src.preprocessing.registry import build_preprocessor
from src.preprocessing.standard import StandardSignalPreprocessor


def test_standard_preprocessor_zscore():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    preprocessor = StandardSignalPreprocessor(
        detrend=False,
        scaler="zscore",
    )

    y = preprocessor(x)

    assert y.dtype == np.float32
    assert abs(float(y.mean())) < 1e-6
    assert abs(float(y.std()) - 1.0) < 1e-5


def test_standard_preprocessor_robust():
    x = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float32)

    preprocessor = StandardSignalPreprocessor(
        detrend=False,
        scaler="robust",
    )

    y = preprocessor(x)

    assert y.dtype == np.float32
    assert np.isfinite(y).all()


def test_legacy_preprocessor_names_still_work():
    for name in ["prep_base", "prep_filter", "prep_domain_norm"]:
        preprocessor = build_preprocessor(name)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = preprocessor(x)

        assert y.dtype == np.float32
        assert np.isfinite(y).all()


def test_standard_registry_name_works():
    preprocessor = build_preprocessor(
        "standard",
        {
            "detrend": False,
            "scaler": "zscore",
            "clip_quantile": None,
        },
    )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = preprocessor(x)

    assert y.dtype == np.float32
    assert np.isfinite(y).all()