#!/usr/bin/env python3
"""Harmonic-template pitch of the guitar range, robust to distortion.

Scores each semitone by how well its harmonic series explains the spectrum,
after subtracting the clip-wide median (the steady backing track).
"""

from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WAV = ROOT / "audio.wav"
SR = 22050
N_FFT = 8192
HOP = 512
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Guitar box we can actually see is roughly A3..G4, but keep a wider net.
MIDI_LO = 52  # E3
MIDI_HI = 76  # E5
WEIGHTS = np.array([1.0, 0.9, 0.75, 0.6, 0.45, 0.35, 0.25, 0.18])


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def midi_name(m: int) -> str:
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == SR
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def stft_mag(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.hanning(N_FFT)
    n = 1 + (len(x) - N_FFT) // HOP
    spec = np.empty((n, N_FFT // 2 + 1), dtype=np.float64)
    for i in range(n):
        frame = x[i * HOP : i * HOP + N_FFT] * window
        spec[i] = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    times = (np.arange(n) * HOP + N_FFT / 2) / SR
    return spec, freqs, times


def harmonic_scores(spec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """spec: frames x bins, already residual (median removed, clipped at 0)."""
    midis = np.arange(MIDI_LO, MIDI_HI + 1)
    scores = np.zeros((len(spec), len(midis)))
    # Precompute bin windows for each harmonic of each midi.
    windows = []
    for m in midis:
        f0 = midi_to_hz(int(m))
        bins = []
        for k, _w in enumerate(WEIGHTS, start=1):
            f = f0 * k
            if f >= freqs[-1] - 30:
                bins.append(None)
                continue
            idx = int(np.argmin(np.abs(freqs - f)))
            lo = max(0, idx - 2)
            hi = min(len(freqs), idx + 3)
            bins.append((lo, hi))
        windows.append(bins)

    log_spec = np.log1p(spec)
    for i, frame in enumerate(log_spec):
        for j, bins in enumerate(windows):
            s = 0.0
            for w, span in zip(WEIGHTS, bins):
                if span is None:
                    continue
                lo, hi = span
                s += w * frame[lo:hi].max()
            scores[i, j] = s
    return scores, midis


def main() -> None:
    x = read_wav(WAV)
    spec, freqs, times = stft_mag(x)
    # Drop energy below 70 Hz (rumble) and above 4 kHz (hiss) before the median,
    # but keep harmonics up to ~3 kHz for the template.
    band = (freqs >= 70) & (freqs <= 3500)
    spec[:, ~band] = 0
    median = np.median(spec, axis=0)
    residual = np.maximum(spec - median, 0)

    scores, midis = harmonic_scores(residual, freqs)
    # Normalize per frame so a loud hit doesn't dominate the margin test.
    peak = scores.max(axis=1, keepdims=True)
    peak[peak < 1e-9] = 1
    norm = scores / peak

    order = np.argsort(scores, axis=1)
    best = order[:, -1]
    second = order[:, -2]
    third = order[:, -3]
    margin = scores[np.arange(len(scores)), best] - scores[np.arange(len(scores)), second]

    out_csv = ROOT / "harmonic_frames.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["time_s", "best", "second", "third", "margin", "best_score", "second_score"]
        )
        for i, t in enumerate(times):
            w.writerow(
                [
                    f"{t:.3f}",
                    midi_name(int(midis[best[i]])),
                    midi_name(int(midis[second[i]])),
                    midi_name(int(midis[third[i]])),
                    f"{margin[i]:.3f}",
                    f"{scores[i, best[i]]:.3f}",
                    f"{scores[i, second[i]]:.3f}",
                ]
            )

    # Collapse into stable events: same best note, margin above a floor, gap < 0.12 s.
    events = []
    i = 0
    n = len(times)
    while i < n:
        if margin[i] < 0.35:
            i += 1
            continue
        j = i + 1
        note = int(midis[best[i]])
        while j < n and int(midis[best[j]]) == note and times[j] - times[j - 1] < 0.12:
            j += 1
        dur = times[j - 1] - times[i]
        if dur >= 0.08 and np.median(margin[i:j]) >= 0.45:
            # second note if it stays close — possible dyad
            sec_counts = {}
            for k in range(i, j):
                name = midi_name(int(midis[second[k]]))
                sec_counts[name] = sec_counts.get(name, 0) + 1
            sec = max(sec_counts, key=sec_counts.get)
            sec_share = sec_counts[sec] / (j - i)
            events.append(
                {
                    "t0": times[i],
                    "t1": times[j - 1],
                    "note": midi_name(note),
                    "midi": note,
                    "median_margin": float(np.median(margin[i:j])),
                    "second": sec if sec_share >= 0.4 else "",
                    "frames": j - i,
                }
            )
        i = j

    ev_path = ROOT / "harmonic_events.csv"
    with ev_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()) if events else ["t0"])
        w.writeheader()
        w.writerows(events)

    # Salience image: time x note, guitar register only.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show_lo, show_hi = 55, 68  # G3..G#4, the visible box plus a bit
    sl = int(np.where(midis == show_lo)[0][0])
    sh = int(np.where(midis == show_hi)[0][0]) + 1
    img = norm[:, sl:sh].T
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=140)
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[times[0], times[-1], show_lo - 0.5, show_hi + 0.5],
        vmin=0.35,
        vmax=1,
    )
    yticks = list(range(show_lo, show_hi + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([midi_name(m) for m in yticks], fontsize=8)
    ax.set_xlabel("секунды")
    ax.set_title("Гармоническая салиентность гитарного регистра (медиана бэка вычтена)")
    for ev in events:
        if show_lo <= ev["midi"] <= show_hi:
            ax.plot([ev["t0"], ev["t1"]], [ev["midi"], ev["midi"]], color="cyan", lw=2)
    fig.tight_layout()
    fig.savefig(ROOT / "salience.png")
    print(f"frames {n} events {len(events)} -> {ev_path.name}")
    for ev in events:
        print(
            f"{ev['t0']:5.2f}-{ev['t1']:5.2f}  {ev['note']:4}  "
            f"margin {ev['median_margin']:.2f}  second {ev['second'] or '-'}"
        )


if __name__ == "__main__":
    main()
