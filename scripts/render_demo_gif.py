"""Render the deterministic README demo GIF.

Requires Pillow. The values shown are rounded from the real vLLM quick-start
evidence already documented in README.md.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 675
BACKGROUND = "#071018"
PANEL = "#0d1726"
PANEL_2 = "#111d2e"
BORDER = "#26364f"
TEXT = "#e7edf5"
MUTED = "#8fa3bb"
GREEN = "#44d49d"
CYAN = "#5bd6e8"
PURPLE = "#a78bfa"
YELLOW = "#f6c85f"
RED = "#ff7b86"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(size: int, *, bold: bool = False):
    return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH, size)


F_TITLE = font(28, bold=True)
F_SUBTITLE = font(18)
F_BODY = font(22)
F_SMALL = font(18)
F_BADGE = font(16, bold=True)
F_RESULT = font(24, bold=True)


def base_frame(scene: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((26, 22, WIDTH - 26, HEIGHT - 22), radius=18, fill=PANEL, outline=BORDER)
    draw.text((54, 42), "JEV / RUNTIME", font=F_TITLE, fill=TEXT)
    draw.text(
        (292, 50),
        "typed decisions on the model server you already run",
        font=F_SUBTITLE,
        fill=MUTED,
    )
    badge(draw, 874, 43, "SGLang native", PURPLE)
    badge(draw, 1030, 43, "vLLM native", CYAN)

    draw.rounded_rectangle(
        (48, 96, WIDTH - 48, HEIGHT - 46), radius=12, fill="#090f1b", outline=BORDER
    )
    draw.ellipse((70, 116, 82, 128), fill=RED)
    draw.ellipse((91, 116, 103, 128), fill=YELLOW)
    draw.ellipse((112, 116, 124, 128), fill=GREEN)
    draw.text((146, 110), scene, font=F_SMALL, fill=MUTED)
    return image, draw


def badge(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, color: str) -> None:
    bounds = draw.textbbox((0, 0), label, font=F_BADGE)
    width = bounds[2] - bounds[0] + 22
    draw.rounded_rectangle((x, y, x + width, y + 28), radius=14, fill="#152236", outline=color)
    draw.text((x + 11, y + 5), label, font=F_BADGE, fill=color)


def line(
    draw: ImageDraw.ImageDraw,
    y: int,
    left: str,
    right: str = "",
    *,
    left_color: str = TEXT,
    right_color: str = TEXT,
) -> None:
    draw.text((76, y), left, font=F_BODY, fill=left_color)
    if right:
        draw.text((392, y), right, font=F_BODY, fill=right_color)


def probability_bar(
    draw: ImageDraw.ImageDraw, y: int, label: str, value: float, color: str
) -> None:
    draw.text((102, y), f"{label:<10}", font=F_SMALL, fill=MUTED)
    left, right = 270, 920
    draw.rounded_rectangle((left, y + 4, right, y + 20), radius=8, fill="#19263a")
    draw.rounded_rectangle(
        (left, y + 4, left + int((right - left) * value), y + 20),
        radius=8,
        fill=color,
    )
    draw.text((944, y - 1), f"{value * 100:5.2f}%", font=F_SMALL, fill=color)


def decision_frame(command: str, phase: int, pulse: bool = False) -> Image.Image:
    image, draw = base_frame("decision.json")
    draw.text((76, 154), "$ ", font=F_BODY, fill=GREEN)
    draw.text((102, 154), command, font=F_BODY, fill=TEXT)
    if phase == 0:
        draw.rectangle((104 + len(command) * 13, 155, 107 + len(command) * 13, 179), fill=CYAN)
        return image

    draw.text((76, 202), "INPUT", font=F_BADGE, fill=PURPLE)
    line(draw, 234, "text", '"I was charged twice and would like a refund."', right_color=TEXT)
    line(draw, 268, "intent", "choice[billing, technical, other]", left_color=CYAN)
    line(draw, 302, "refund_requested", "boolean", left_color=CYAN)
    if phase == 1:
        return image

    if phase == 2:
        color = CYAN if pulse else PURPLE
        draw.text((76, 354), "scoring selected labels", font=F_BODY, fill=color)
        draw.text((378, 354), "·  ·  ·", font=F_BODY, fill=MUTED)
        return image

    draw.text((76, 350), "TYPED RESULT", font=F_BADGE, fill=GREEN)
    line(draw, 382, "intent  [choice]", "billing", left_color=CYAN, right_color=GREEN)
    if phase == 3:
        return image
    probability_bar(draw, 420, "billing", 0.7159, GREEN)
    probability_bar(draw, 448, "technical", 0.1597, PURPLE)
    probability_bar(draw, 476, "other", 0.1244, YELLOW)
    if phase == 4:
        return image

    line(draw, 516, "refund_requested", "true", left_color=CYAN, right_color=GREEN)
    draw.text((655, 520), "true 79.82%  ·  false 20.18%", font=F_SMALL, fill=TEXT)
    if phase == 5:
        return image

    draw.rounded_rectangle((72, 566, WIDTH - 72, 614), radius=8, fill=PANEL_2)
    draw.text((92, 579), "bundle default@1", font=F_SMALL, fill=PURPLE)
    draw.text((390, 579), "generation 1", font=F_SMALL, fill=CYAN)
    draw.text((650, 579), "engine vLLM", font=F_SMALL, fill=TEXT)
    draw.text((900, 579), "answered 2/2", font=F_SMALL, fill=GREEN)
    return image


def switch_frame(phase: int) -> Image.Image:
    image, draw = base_frame("live bundle update")
    draw.text((76, 158), '$ jevctl bundle prepare demo@2 --url "$JEV_URL"', font=F_BODY, fill=TEXT)
    if phase >= 1:
        draw.text((102, 204), "✓ canary passed", font=F_BODY, fill=GREEN)
    if phase >= 2:
        draw.text(
            (76, 254),
            '$ jevctl bundle activate demo@2 decision-model 1 --url "$JEV_URL"',
            font=F_SMALL,
            fill=TEXT,
        )
    if phase >= 3:
        draw.rounded_rectangle((76, 302, WIDTH - 76, 380), radius=10, fill=PANEL_2, outline=GREEN)
        draw.text((100, 320), "ACTIVE", font=F_BADGE, fill=GREEN)
        draw.text((222, 316), "default@1", font=F_RESULT, fill=MUTED)
        draw.text((400, 316), "→", font=F_RESULT, fill=CYAN)
        draw.text((455, 316), "demo@2", font=F_RESULT, fill=PURPLE)
        draw.text((780, 320), "generation 1 → 2", font=F_SMALL, fill=CYAN)
        draw.text((100, 352), "atomic publication · no model reload", font=F_SMALL, fill=MUTED)
    if phase >= 4:
        draw.text(
            (76, 420),
            '$ jevctl decide examples/request.json --url "$JEV_URL"',
            font=F_BODY,
            fill=TEXT,
        )
        draw.rounded_rectangle((76, 468, WIDTH - 76, 575), radius=10, fill="#10201e", outline=GREEN)
        draw.text((100, 488), "✓ billing", font=F_RESULT, fill=GREEN)
        draw.text((330, 493), "p=0.7159", font=F_SMALL, fill=TEXT)
        draw.text((548, 488), "✓ refund_requested: true", font=F_RESULT, fill=GREEN)
        draw.text((100, 535), "bundle demo@2", font=F_SMALL, fill=PURPLE)
        draw.text((390, 535), "generation 2", font=F_SMALL, fill=CYAN)
        draw.text((650, 535), "request version pinned", font=F_SMALL, fill=TEXT)
    if phase >= 5:
        draw.text(
            (330, 603),
            "typed · probabilistic · versioned",
            font=F_RESULT,
            fill=TEXT,
        )
    return image


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "jev-demo.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(decision_frame("", 0))
    durations.append(1500)
    command = 'jevctl decide examples/request.json --url "$JEV_URL"'
    for length in (7, 14, 21, 28, 35, 42, 49, len(command)):
        frames.append(decision_frame(command[:length], 0))
        durations.append(120)
    frames.append(decision_frame(command, 1))
    durations.append(2000)
    for pulse in (False, True, False, True, False, True):
        frames.append(decision_frame(command, 2, pulse))
        durations.append(250)
    for phase, duration in ((3, 2000), (4, 2000), (5, 2000), (6, 2400)):
        frames.append(decision_frame(command, phase))
        durations.append(duration)
    for phase, duration in ((0, 1800), (1, 1800), (2, 1800), (3, 2800), (4, 4500), (5, 2940)):
        frames.append(switch_frame(phase))
        durations.append(duration)

    assert sum(durations) == 30_000
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {output} ({len(frames)} frames, {sum(durations) / 1000:.0f}s)")


if __name__ == "__main__":
    main()
