"""Connect image -> detector -> severity engine -> SQLite."""
import sys
from pathlib import Path
import cv2

# Supports running from the repository root or importing src.pipeline.
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from severity import evaluate, load_config
from storage import save_result


def draw_zone(image_bgr, zone):
    height, width = image_bgr.shape[:2]
    output = image_bgr.copy()
    cv2.rectangle(
        output,
        (int(zone[0] * width), int(zone[1] * height)),
        (int(zone[2] * width), int(zone[3] * height)),
        (0, 165, 255), 1,
    )
    return output


def inspect(image_bgr, image_name, cfg=None, profile="Standard", save=True, detector_fn=None):
    """
    detector_fn must follow Member A's contract:
    detect(image_bgr, min_conf=...) -> (defects, annotated_image, inference_ms).
    Passing detector_fn explicitly makes unit/integration testing easier.
    """
    if detector_fn is None:
        # Import only when needed, so severity/storage can be tested before A's file arrives.
        from detector import detect as detector_fn

    cfg = cfg or load_config(profile=profile)
    defects, annotated, inference_ms = detector_fn(
        image_bgr, min_conf=cfg["min_confidence"]
    )
    result = evaluate(defects, cfg)
    result["annotated"] = draw_zone(annotated, cfg["critical_zone"])
    result["inference_ms"] = inference_ms
    result["id"] = save_result(image_name, result, profile) if save else None
    return result
