"""Chinese number hand-gesture classifier.

This project intentionally uses geometry rather than raw landmark coordinates.
That makes the classifier substantially less sensitive to camera position and
hand tilt than rules such as "tip_y < joint_y".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import math
import numpy as np

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "thumb": (THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

NUMBER_INFO: Dict[int, Tuple[str, str, str]] = {
    1: ("一", "yī", "index"),
    2: ("二", "èr", "index + middle"),
    3: ("三", "sān", "thumb + index + middle"),
    4: ("四", "sì", "index + middle + ring + pinky"),
    5: ("五", "wǔ", "open hand"),
    6: ("六", "liù", "thumb + pinky"),
    7: ("七", "qī", "thumb + index + middle pinch"),
    8: ("八", "bā", "thumb + index"),
    9: ("九", "jiǔ", "hooked index"),
    10: ("十", "shí", "fist"),
}

@dataclass
class Features:
    scale: float
    extended: Dict[str, bool]
    tip_dist: Dict[str, float]
    thumb_index: float
    thumb_middle: float
    index_hook: float
    palm_open: float

def _p(lm: Sequence, i: int) -> np.ndarray:
    return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=float)

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / denom, -1, 1))))

def _finger_extension(lm, mcp, pip, tip, scale) -> bool:
    wrist = _p(lm, WRIST)
    m = _p(lm, mcp)
    p = _p(lm, pip)
    t = _p(lm, tip)
    # A finger is extended when its tip is substantially farther from the
    # wrist than its PIP and the finger itself is relatively straight.
    radial = dist(t, wrist) > dist(p, wrist) + 0.18 * scale
    straight = angle(m, p, t) > 145
    return radial and straight

def extract_features(lm: Sequence) -> Features:
    wrist = _p(lm, WRIST)
    palm_center = np.mean([_p(lm, i) for i in (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)], axis=0)
    scale = max(dist(wrist, palm_center), 1e-5)

    extended = {
        "index": _finger_extension(lm, INDEX_MCP, INDEX_PIP, INDEX_TIP, scale),
        "middle": _finger_extension(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, scale),
        "ring": _finger_extension(lm, RING_MCP, RING_PIP, RING_TIP, scale),
        "pinky": _finger_extension(lm, PINKY_MCP, PINKY_PIP, PINKY_TIP, scale),
    }

    thumb_tip = _p(lm, THUMB_TIP)
    index_tip = _p(lm, INDEX_TIP)
    middle_tip = _p(lm, MIDDLE_TIP)

    # Thumb extension is more reliable from CMC-to-tip reach than y/x tests.
    thumb_extended = dist(thumb_tip, wrist) > dist(_p(lm, THUMB_IP), wrist) + 0.12 * scale
    extended["thumb"] = thumb_extended

    tip_dist = {
        "thumb-index": dist(thumb_tip, index_tip) / scale,
        "thumb-middle": dist(thumb_tip, middle_tip) / scale,
        "index-middle": dist(index_tip, middle_tip) / scale,
    }

    idx_mcp, idx_pip, idx_dip, idx_tip = map(lambda i: _p(lm, i),
                                               (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP))
    # 0 means straight, 1 means strongly hooked.
    idx_angle = angle(idx_mcp, idx_pip, idx_dip)
    dip_angle = angle(idx_pip, idx_dip, idx_tip)
    index_hook = float(np.clip(((150 - idx_angle) + (150 - dip_angle)) / 180, 0, 1))

    palm_open = sum(extended.values()) / 5.0
    return Features(
        scale=scale,
        extended=extended,
        tip_dist=tip_dist,
        thumb_index=tip_dist["thumb-index"],
        thumb_middle=tip_dist["thumb-middle"],
        index_hook=index_hook,
        palm_open=palm_open,
    )

def _score(expected: Dict[str, bool], actual: Dict[str, bool]) -> float:
    matches = sum(expected[k] == actual[k] for k in expected)
    return matches / len(expected)

def classify(lm: Sequence) -> Tuple[Optional[int], float]:
    """Return (number, confidence). Confidence is heuristic, not ML probability."""
    f = extract_features(lm)
    e = f.extended

    # Closed hand is the simplest and most stable 10.
    if not any(e.values()) and f.palm_open < 0.22:
        return 10, 0.96

    # 5: all five digits extended.
    if all(e.values()):
        return 5, 0.97

    # 6: thumb + pinky, middle three folded.
    if e["thumb"] and e["pinky"] and not e["index"] and not e["middle"] and not e["ring"]:
        return 6, 0.94

    # 8: thumb + index in an L/gun-like pose.
    if e["thumb"] and e["index"] and not e["middle"] and not e["ring"] and not e["pinky"]:
        if f.thumb_index > 0.35:
            return 8, min(0.98, 0.78 + f.thumb_index / 3.0)

    # 9: curled/hooked index with other fingers closed.
    if f.index_hook > 0.62 and not e["middle"] and not e["ring"] and not e["pinky"]:
        return 9, min(0.97, 0.75 + f.index_hook * 0.22)

    # 7: thumb, index and middle fingertips gathered together.
    if e["thumb"] and e["index"] and e["middle"] and not e["ring"] and not e["pinky"]:
        pinch = max(0.0, 1.0 - min(f.thumb_index, f.thumb_middle) / 0.55)
        if f.thumb_index < 0.60 or f.thumb_middle < 0.60:
            return 7, min(0.96, 0.72 + pinch * 0.24)

    # 4: four fingers extended, thumb folded.
    if not e["thumb"] and e["index"] and e["middle"] and e["ring"] and e["pinky"]:
        return 4, 0.95

    # 3: common mainland variant: thumb + index + middle.
    if e["thumb"] and e["index"] and e["middle"] and not e["ring"] and not e["pinky"]:
        return 3, 0.93

    if e["index"] and e["middle"] and not e["ring"] and not e["pinky"] and not e["thumb"]:
        return 2, 0.96

    if e["index"] and not e["middle"] and not e["ring"] and not e["pinky"] and not e["thumb"]:
        return 1, 0.95

    return None, 0.0
