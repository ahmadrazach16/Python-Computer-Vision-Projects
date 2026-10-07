from __future__ import annotations
from pathlib import Path
from typing import Optional
import cv2
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
    "/System/Library/Fonts/PingFang.ttc",
]

def find_cjk_font() -> Optional[str]:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None

class ChineseOverlay:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.font_path = find_cjk_font()
        self.font_big = ImageFont.truetype(self.font_path, 92) if self.font_path else None
        self.font_mid = ImageFont.truetype(self.font_path, 42) if self.font_path else None
        self.font_small = ImageFont.truetype(self.font_path, 25) if self.font_path else None

    def render(self, frame, number, chinese, pinyin, confidence, fps, audio_on):
        # OpenCV draws the camera/landmark layer; Pillow draws CJK text.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im, "RGBA")

        panel = (20, 20, 20, 210)
        draw.rounded_rectangle((25, 25, 430, 250), radius=25, fill=panel)

        if number is not None:
            draw.text((52, 45), str(number), font=self.font_big, fill=(255,255,255,255))
            if self.font_mid:
                draw.text((155, 58), chinese, font=self.font_mid, fill=(100,210,255,255))
                draw.text((155, 115), pinyin, font=self.font_mid, fill=(255,255,255,255))
            else:
                draw.text((155, 65), "Chinese font not found", font=self.font_small, fill=(255,220,120,255))
            draw.text((52, 180), f"confidence  {confidence:.0%}", font=self.font_small, fill=(210,220,230,255))
        else:
            draw.text((52, 65), "Show 1–10", font=self.font_mid, fill=(255,255,255,255))
            draw.text((52, 125), "Chinese hand counter", font=self.font_small, fill=(180,200,220,255))

        draw.rounded_rectangle((25, self.height-65, 430, self.height-25), radius=15, fill=(20,20,20,180))
        draw.text((42, self.height-57), f"FPS {fps:.0f}   |   Audio {'ON' if audio_on else 'OFF'}", font=self.font_small, fill=(235,235,235,255))

        return cv2.cvtColor(__import__("numpy").array(im), cv2.COLOR_RGB2BGR)
