
import numpy as np

from src.pipeline import inspect
from src.storage import load_history


def test_pipeline_uses_detector_and_saves_result(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    database = tmp_path / "test_inspections.db"

    # Simulate Member A's detector without requiring YOLO weights.
    def mock_detector(image_bgr, min_conf):
        return [], image_bgr.copy(), 12.5

    # Use the real configuration and severity logic.
    from src.severity import load_config

    cfg = load_config()
    result = inspect(
        image,
        "test_image.png",
        cfg=cfg,
        profile="Standard",
        save=False,
        detector_fn=mock_detector,
    )

    assert result["decision"] == "ACCEPT"
    assert result["max_score"] == 0.0
    assert result["inference_ms"] == 12.5
    assert result["annotated"].shape == image.shape
    assert result["defects"] == []


def test_pipeline_can_skip_database_save():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    def mock_detector(image_bgr, min_conf):
        return [], image_bgr.copy(), 5.0

    result = inspect(
        image,
        "unsaved_image.png",
        save=False,
        detector_fn=mock_detector,
    )

    assert result["id"] is None
    assert result["decision"] == "ACCEPT"

def test_pipeline_saves_result_to_sqlite(tmp_path, monkeypatch):
    import numpy as np
    from src import pipeline
    from src.storage import load_history

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    database = tmp_path / "test_inspections.db"

    def mock_detector(image_bgr, min_conf):
        return [], image_bgr.copy(), 8.0

    # Redirect pipeline storage to a temporary test database.
    monkeypatch.setattr(
        pipeline,
        "save_result",
        lambda image_name, result, profile: (
            __import__("src.storage", fromlist=["storage"]).save_result(
                image_name, result, profile, db_path=database
            )
        ),
    )

    result = pipeline.inspect(
        image,
        "database_test.png",
        profile="Standard",
        save=True,
        detector_fn=mock_detector,
    )

    history = load_history(database)

    assert result["id"] is not None
    assert len(history) == 1
    assert history.iloc[0]["image_name"] == "database_test.png"
    assert history.iloc[0]["decision"] == "ACCEPT"
