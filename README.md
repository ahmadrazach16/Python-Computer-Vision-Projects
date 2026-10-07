<<<<<<< HEAD
# Chinese Hand Counter 🇨🇳✋

A webcam-based Computer Vision project that recognizes **Chinese number hand gestures from 1 to 10** and displays:

- Arabic number: `1–10`
- Chinese character: `一 二 三 四 五 六 七 八 九 十`
- Pinyin: `yī, èr, sān, sì, wǔ, liù, qī, bā, jiǔ, shí`
- Mandarin speech when the stable prediction changes

## Demo concept

```text
Webcam
   ↓
MediaPipe Hands
   ↓
21 hand landmarks
   ↓
Geometry / distance features
   ↓
Chinese gesture rules
   ↓
7-frame temporal smoothing
   ↓
Number + Chinese character + Pinyin
   ↓
Optional Mandarin TTS
```

## Why this approach?

MediaPipe gives the hand landmarks, but **the gesture semantics are project logic**.

This implementation deliberately avoids brittle rules such as:

> "Is this fingertip above that fingertip?"

Instead, it normalizes distances using the palm scale and uses finger extension, joint angles, fingertip distances, and a few shape-specific rules. That makes the system more tolerant of hand size, camera distance, and moderate rotation.

## Gesture set

This project uses a common mainland-Chinese style:

| # | Character | Pinyin | Gesture |
|---|---|---|---|
| 1 | 一 | yī | index |
| 2 | 二 | èr | index + middle |
| 3 | 三 | sān | thumb + index + middle |
| 4 | 四 | sì | four fingers, thumb folded |
| 5 | 五 | wǔ | open hand |
| 6 | 六 | liù | thumb + pinky |
| 7 | 七 | qī | thumb + index + middle fingertips together |
| 8 | 八 | bā | thumb + index |
| 9 | 九 | jiǔ | hooked index |
| 10 | 十 | shí | closed fist |

Chinese number gestures vary by region, especially 7–10. The classifier therefore targets one documented/common convention rather than claiming a universal mapping.

## Install

### Windows

```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd chinese-hand-counter

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

With Mandarin audio:

```powershell
python main.py
```

Without audio:

```powershell
python main.py --no-audio
```

If your webcam is not camera 0:

```powershell
python main.py --camera 1
```

Press **Q** or **ESC** to exit.

## Mandarin audio

Audio uses `pyttsx3`, which relies on the operating system's installed speech engine.

If a Mandarin/Chinese voice is installed, the app will prefer it automatically. If no Chinese voice is available, recognition still works and the application does not crash.

On Windows, add a Chinese/Mandarin speech voice through:

**Settings → Time & language → Speech**

## Chinese text rendering

OpenCV's standard text renderer does not reliably render Chinese characters. The UI therefore uses **Pillow** for the text layer.

The application automatically looks for common CJK fonts such as:

- Microsoft YaHei
- SimHei
- SimSun
- Noto Sans CJK
- PingFang

## Flicker reduction

Raw webcam predictions can jump between neighboring gestures. A seven-frame majority vote is applied before the UI changes its stable number. Speech also has a cooldown so it does not repeat on every frame.

## Project structure

```text
chinese-hand-counter/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── gestures.py
│   ├── stability.py
│   ├── speech.py
│   └── ui.py
├── tests/
│   └── test_gestures.py
└── assets/
```

## Engineering notes

### 21 landmarks

MediaPipe Hands provides 21 landmarks:

```text
wrist
thumb: CMC → MCP → IP → tip
index: MCP → PIP → DIP → tip
middle: MCP → PIP → DIP → tip
ring: MCP → PIP → DIP → tip
pinky: MCP → PIP → DIP → tip
```

### Geometry instead of raw coordinates

Every hand is different and the same hand can appear at different distances from the camera. The classifier uses the wrist/palm geometry as a scale reference, then compares normalized distances and angles.

### Hard gestures

The project pays special attention to:

- **7** — fingertips gathered together
- **9** — hooked index finger
- **10** — closed fist

These shapes are more ambiguous than simple 1–5 finger extension and benefit most from temporal smoothing.

## Limitations

This is a rule-based educational/portfolio Computer Vision system, not a trained gesture-recognition neural network.

Performance can decrease with:

- severe occlusion
- very low light
- motion blur
- extreme hand rotation
- multiple hands
- gestures outside the selected regional convention

## Future upgrades

- Add a learned classifier trained on landmark vectors
- Support both left and right hand with calibrated orientation
- Add a confidence history graph
- Record gesture datasets directly from the webcam
- Add 0 and alternative 10 gestures
- Add Chinese pronunciation waveform/audio feedback
- Build a Streamlit/web version
- Export recognition events to CSV

## License

MIT
=======
# Python-Computer-vision-Projects
open cv
>>>>>>> 6d970a95af5f3e6295bd3240b5e95005f3b85f4f
