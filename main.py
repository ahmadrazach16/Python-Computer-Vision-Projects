from __future__ import annotations

import argparse
import time
import cv2
import mediapipe as mp

from src.gestures import NUMBER_INFO, classify
from src.speech import ChineseSpeaker
from src.stability import PredictionSmoother
from src.ui import ChineseOverlay

def parse_args():
    p = argparse.ArgumentParser(description="Chinese 1–10 webcam hand-gesture counter")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--no-audio", action="store_true", help="Disable Mandarin TTS")
    p.add_argument("--min-detection", type=float, default=0.60)
    p.add_argument("--min-tracking", type=float, default=0.60)
    return p.parse_args()

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        model_complexity=1,
    )

    smoother = PredictionSmoother(window=7, min_votes=4)
    speaker = ChineseSpeaker(enabled=not args.no_audio)
    overlay = None

    prev = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            if overlay is None:
                h, w = frame.shape[:2]
                overlay = ChineseOverlay(w, h)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            raw_label, raw_conf = None, 0.0
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                raw_label, raw_conf = classify(hand.landmark)

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(80, 220, 255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 170, 80), thickness=2),
                )

            stable = smoother.update(raw_label, raw_conf)

            number = stable.label
            chinese = pinyin = ""
            if number in NUMBER_INFO:
                chinese, pinyin, _ = NUMBER_INFO[number]
                if stable.changed:
                    speaker.speak(chinese)

            now = time.perf_counter()
            instant = 1.0 / max(now - prev, 1e-6)
            fps = 0.9 * fps + 0.1 * instant if fps else instant
            prev = now

            frame = overlay.render(
                frame, number, chinese, pinyin,
                stable.confidence if number is not None else 0.0,
                fps, not args.no_audio
            )

            cv2.putText(
                frame, "Q / ESC  =  quit",
                (frame.shape[1] - 235, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230,230,230), 1, cv2.LINE_AA
            )

            cv2.imshow("Chinese Hand Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        speaker.close()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

