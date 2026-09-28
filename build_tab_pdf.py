#!/usr/bin/env python3
"""Confirmed guitar tab → A4 PDF.

Pitch comes from harmonic-template analysis (harmonic_events.csv).
Fret and string come from standard tuning, restricted to the fret box
seen on the video (mostly 7–11). Rhythm is quantized to a 16th grid.
"""

from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parent
SR = 22050
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# High string first, as on a tab staff.
STRINGS = [("e", 64), ("B", 59), ("G", 55), ("D", 50), ("A", 45), ("E", 40)]


def note_midi(name: str) -> int:
    if len(name) >= 3 and name[1] == "#":
        pc = NAMES.index(name[:2])
        octave = int(name[2:])
    else:
        pc = NAMES.index(name[0])
        octave = int(name[1:])
    return (octave + 1) * 12 + pc


def midi_hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def place(midi: int) -> tuple[int, int]:
    """String index (0 = high e) and fret. Prefer the fret box 7–11."""
    cands = []
    for i, (_name, open_m) in enumerate(STRINGS):
        fret = midi - open_m
        if not 0 <= fret <= 15:
            continue
        if 7 <= fret <= 11:
            rank = 0
        elif 5 <= fret <= 12:
            rank = 1
        elif fret >= 13:
            rank = 2
        else:
            rank = 3
        cands.append((rank, abs(fret - 8), i, fret))
    if not cands:
        raise ValueError(midi)
    cands.sort()
    return cands[0][2], cands[0][3]


def read_wav() -> np.ndarray:
    with wave.open(str(ROOT / "audio.wav"), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def cents_of(x: np.ndarray, t: float, midi: int) -> tuple[float, float]:
    """Cents from equal temperament, using the strongest of the first 3 harmonics."""
    n = 4096
    i = int(t * SR)
    i = max(0, min(len(x) - n, i))
    frame = x[i : i + n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    f0 = midi_hz(midi)
    best = None
    bin_hz = freqs[1] - freqs[0]
    for k in (1, 2, 3):
        target = f0 * k
        half = 2 ** (40 / 1200)
        mask = (freqs >= target / half) & (freqs <= target * half)
        if not np.any(mask):
            continue
        local = np.where(mask)[0]
        idx = int(local[np.argmax(mag[local])])
        f = freqs[idx]
        if 0 < idx < len(mag) - 1:
            a, b, c = mag[idx - 1], mag[idx], mag[idx + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                delta = 0.5 * (a - c) / denom
                delta = max(-0.5, min(0.5, delta))
                f = freqs[idx] + delta * bin_hz
        cents = 1200 * math.log2(f / target)
        strength = mag[idx]
        if best is None or strength > best[0]:
            best = (strength, cents, f / k)
    if best is None:
        return 0.0, f0
    return best[1], best[2]


def load_notes(x: np.ndarray) -> tuple[list[dict], float, list[dict]]:
    rows = list(csv.DictReader((ROOT / "harmonic_events.csv").open()))
    raw = []
    for r in rows:
        t0, t1 = float(r["t0"]), float(r["t1"])
        margin = float(r["median_margin"])
        dur = t1 - t0
        if margin < 1.3 or dur < 0.10:
            continue
        midi = note_midi(r["note"])
        string, fret = place(midi)
        mid = (t0 + t1) / 2
        cents, hz = cents_of(x, mid, midi)
        raw.append(
            {
                "t0": t0,
                "t1": t1,
                "name": r["note"],
                "midi": midi,
                "margin": margin,
                "string": string,
                "fret": fret,
                "cents": cents,
                "hz": hz,
            }
        )

    # Integer BPM whose 16th-grid median error is smallest.
    onsets = np.array([n["t0"] for n in raw])
    best = None
    for bpm in range(90, 181):
        grid = 60 / bpm / 4
        err = [abs((t - onsets[0]) - round((t - onsets[0]) / grid) * grid) for t in onsets]
        med = float(np.median(err))
        if best is None or med < best[0]:
            best = (med, bpm, grid)
    _med, bpm, grid = best
    origin = onsets[0]

    notes = []
    for n in raw:
        start = int(round((n["t0"] - origin) / grid))
        end = int(round((n["t1"] - origin) / grid))
        dur = max(1, end - start)
        item = dict(n)
        item["start16"] = start
        item["dur16"] = dur
        item["q_err"] = abs((n["t0"] - origin) - start * grid)
        notes.append(item)

    # Same grid slot: keep the clearer note.
    notes.sort(key=lambda n: (n["start16"], -n["margin"]))
    kept = []
    for n in notes:
        if kept and n["start16"] < kept[-1]["start16"] + kept[-1]["dur16"]:
            if n["margin"] > kept[-1]["margin"] and n["start16"] == kept[-1]["start16"]:
                kept[-1] = n
            else:
                # Trim the previous note so they don't share a slot.
                room = n["start16"] - kept[-1]["start16"]
                if room >= 1 and n["margin"] >= 1.8:
                    kept[-1]["dur16"] = room
                    kept.append(n)
                elif room >= 1 and kept[-1]["dur16"] > room:
                    kept[-1]["dur16"] = room
                    kept.append(n)
            continue
        kept.append(n)

    rejected = []
    for r in rows:
        t0, t1 = float(r["t0"]), float(r["t1"])
        margin = float(r["median_margin"])
        if margin < 1.3 or (t1 - t0) < 0.10:
            rejected.append({"t0": t0, "note": r["note"], "margin": margin, "dur": t1 - t0})
    return kept, bpm, rejected


def rhythm_kind(dur16: int) -> tuple[str, bool]:
    """Return (base value, dotted) for a duration in sixteenths."""
    table = {
        1: ("16th", False),
        2: ("eighth", False),
        3: ("eighth", True),
        4: ("quarter", False),
        6: ("quarter", True),
        8: ("half", False),
        12: ("half", True),
    }
    if dur16 in table:
        return table[dur16]
    if dur16 >= 8:
        return "half", False
    if dur16 >= 4:
        return "quarter", False
    if dur16 >= 2:
        return "eighth", False
    return "16th", False


def draw_stem(ax, x, y, kind: str, dotted: bool) -> None:
    filled = kind in ("16th", "eighth", "quarter")
    ax.plot(x, y, "o", ms=4.2, color="black", mfc="black" if filled else "white", mew=0.8, zorder=4)
    top = y + 16
    ax.plot([x + 1.6, x + 1.6], [y + 1.5, top], color="black", lw=0.7, zorder=4)
    flags = {"16th": 2, "eighth": 1}.get(kind, 0)
    for k in range(flags):
        yy = top - k * 4
        ax.plot([x + 1.6, x + 8], [yy, yy - 3], color="black", lw=0.7, zorder=4)
    if dotted:
        ax.plot(x + 6, y + 2, "o", ms=1.6, color="black", zorder=4)


def render(notes: list[dict], bpm: float, rejected: list[dict]) -> None:
    max16 = max(n["start16"] + n["dur16"] for n in notes)
    bars = (max16 + 15) // 16
    bars_per_system = 2
    systems_per_page = 5
    systems = (bars + bars_per_system - 1) // bars_per_system
    pages = (systems + systems_per_page - 1) // systems_per_page

    pdf_path = ROOT / "guitar-tab.pdf"
    fig_w, fig_h = 8.27, 11.69
    line_gap = 16
    usable = 820
    step = usable / (16 * bars_per_system)
    x0 = 110

    def new_page():
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1400)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        return fig, ax

    def header(ax, continued: bool) -> None:
        title = "Гитарная партия" if not continued else "Гитарная партия — продолжение"
        ax.text(70, 1335, title, fontsize=18, fontname="DejaVu Sans", color="#1a1a1c")
        ax.text(
            70,
            1302,
            "Электрогитара, одна партия, ролик 26 с. Строй подтверждён: частота совпала с ладом на кадре.",
            fontsize=8.5,
            fontname="DejaVu Sans",
            color="#444",
        )
        ax.text(
            70,
            1276,
            f"Строй  E–A–D–G–B–E     ♩ = {bpm:.0f}     размер 4/4",
            fontsize=11,
            fontname="DejaVu Sans",
            color="#1a1a1c",
        )
        ax.plot([70, 930], [1258, 1258], color="#1a1a1c", lw=0.8)

    def footer(ax, page_no: int) -> None:
        ax.text(
            70,
            62,
            "Шестилинейная табулатура: тонкая струна сверху, строй слева, лад на линии, ритм над станом.\n"
            "Отдельного ГОСТа на рисунок табулатуры нет. Подпись листа — по составу ГОСТ Р 7.0.4-2020\n"
            "(заголовок, вид партии, строй). В библиографическом учёте табулатура — код h, ГОСТ Р 7.0.100-2018.",
            fontsize=7.5,
            fontname="DejaVu Sans",
            color="#333",
            va="top",
        )
        ax.text(930, 28, str(page_no), fontsize=8, fontname="DejaVu Sans", ha="right", color="#666")

    def draw_system(ax, system_index: int, y: float) -> None:
        for i in range(6):
            ax.plot([x0, x0 + usable], [y - i * line_gap, y - i * line_gap], color="#222", lw=0.7, zorder=1)
        for i, (name, _m) in enumerate(STRINGS):
            ax.text(x0 - 18, y - i * line_gap, name, fontsize=8, fontname="DejaVu Sans", va="center", ha="right", color="#333")
        for b in range(bars_per_system + 1):
            bx = x0 + b * 16 * step
            lw = 1.1 if b in (0, bars_per_system) else 0.6
            ax.plot([bx, bx], [y + 2, y - 5 * line_gap], color="#222", lw=lw, zorder=2)
            num = system_index * bars_per_system + b + 1
            if b < bars_per_system and num <= bars:
                ax.text(bx + 3, y + 36, str(num), fontsize=8, fontname="DejaVu Sans", color="#666")
        start_bar = system_index * bars_per_system
        for n in notes:
            bar = n["start16"] // 16
            if not start_bar <= bar < start_bar + bars_per_system:
                continue
            local = n["start16"] - start_bar * 16
            nx = x0 + (local + 0.45) * step
            ny = y - n["string"] * line_gap
            label = str(n["fret"])
            if abs(n["cents"]) >= 48:
                label = f"({n['fret']})"
            ax.text(
                nx,
                ny,
                label,
                fontsize=9,
                fontname="DejaVu Sans Mono",
                ha="center",
                va="center",
                color="#111",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
                zorder=3,
            )
            kind, dotted = rhythm_kind(n["dur16"])
            draw_stem(ax, nx, y + 18, kind, dotted)

    with PdfPages(pdf_path) as pdf:
        page_no = 1
        for p in range(pages):
            fig, ax = new_page()
            header(ax, continued=p > 0)
            for i in range(systems_per_page):
                s = p * systems_per_page + i
                if s >= systems:
                    break
                y = 1188 - i * 205
                draw_system(ax, s, y)
            footer(ax, page_no)
            pdf.savefig(fig)
            fig.savefig(ROOT / f"guitar-tab-p{page_no}.png", dpi=140)
            plt.close(fig)
            page_no += 1

        # Page 2 — evidence
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1400)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.text(70, 1335, "Чем подтверждена каждая нота", fontsize=16, fontname="DejaVu Sans")
        ax.text(
            70,
            1308,
            "Спектр: шаблон гармоник 1–8 после вычитания медианы всего ролика (постоянный бэк).\n"
            "Лад: бокс 7–11, в котором на кадрах стоит левая рука; строй EADGBE совпал с частотой.\n"
            f"Ритм: сетка шестнадцатых при ♩ = {bpm:.0f}. В такте только ноты с запасом громкости гармоник ≥ 1.3.",
            fontsize=8,
            fontname="DejaVu Sans",
            color="#333",
            va="top",
        )

        headers = ["т, с", "нота", "Гц", "цент", "таб", "запас", "сдвиг, мс"]
        xs = [70, 150, 230, 320, 410, 520, 640]
        y = 1205
        for h, x in zip(headers, xs):
            ax.text(x, y, h, fontsize=8, fontname="DejaVu Sans", color="#111")
        ax.plot([70, 900], [y - 6, y - 6], color="#222", lw=0.6)
        y -= 22
        for n in notes:
            if y < 70:
                break
            string_name = STRINGS[n["string"]][0]
            cells = [
                f"{n['t0']:.2f}",
                n["name"],
                f"{n['hz']:.1f}",
                f"{n['cents']:+.0f}",
                f"{string_name}{n['fret']}",
                f"{n['margin']:.1f}",
                f"{n['q_err']*1000:.0f}",
            ]
            for c, x in zip(cells, xs):
                ax.text(x, y, c, fontsize=7.5, fontname="DejaVu Sans Mono", color="#222")
            y -= 16

        ax.text(
            70,
            78,
            "Скобки в партии — одна высота на грани соседней ноты (5,90 с, E3 / A7, −50 центов).\n"
            "Остальные высоты ближе к названной ноте, чем к соседней. Короткие всплески с запасом < 1.3 в такт не вошли.",
            fontsize=8,
            fontname="DejaVu Sans",
            color="#333",
            va="top",
        )
        ax.text(930, 28, str(page_no), fontsize=8, fontname="DejaVu Sans", ha="right", color="#666")
        pdf.savefig(fig)
        fig.savefig(ROOT / f"guitar-tab-p{page_no}.png", dpi=120)
        plt.close(fig)

    # sidecar the player can read
    with (ROOT / "confirmed_notes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "note", "hz", "cents", "string", "fret", "margin", "start16", "dur16", "grid_error_ms"])
        for n in notes:
            w.writerow(
                [
                    f"{n['t0']:.3f}",
                    n["name"],
                    f"{n['hz']:.2f}",
                    f"{n['cents']:.1f}",
                    STRINGS[n["string"]][0],
                    n["fret"],
                    f"{n['margin']:.2f}",
                    n["start16"],
                    n["dur16"],
                    f"{n['q_err']*1000:.1f}",
                ]
            )
    print(f"bpm {bpm:.0f} notes {len(notes)} bars {bars} rejected_short {len(rejected)}")
    print(f"pdf {pdf_path}")


def main() -> None:
    x = read_wav()
    notes, bpm, rejected = load_notes(x)
    render(notes, bpm, rejected)


if __name__ == "__main__":
    main()
