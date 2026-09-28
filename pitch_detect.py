#!/usr/bin/env python3
"""Improved pitch/onset analysis for distorted electric guitar + summary."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

SR = 22050
WIN_MS = 40
HOP_MS = 10
FMIN = 75.0
FMAX = 800.0
YIN_THRESH = 0.20
RMS_SILENCE = 0.008

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note(midi: float) -> str:
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def freq_to_midi(f: float) -> float:
    return 69.0 + 12.0 * math.log2(f / 440.0)


def read_wav(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == SR
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def lowpass(x: np.ndarray, cutoff: float = 900.0) -> np.ndarray:
    # simple 1-pole IIR
    rc = 1.0 / (2 * math.pi * cutoff)
    dt = 1.0 / SR
    a = dt / (rc + dt)
    y = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev = prev + a * (v - prev)
        y[i] = prev
    return y


def yin_pitch(frame: np.ndarray, sr: int = SR):
    n = len(frame)
    x = frame - np.mean(frame)
    x = x * np.hanning(n)
    tau_min = max(2, int(sr / FMAX))
    tau_max = min(n // 2 - 1, int(sr / FMIN))
    if tau_max <= tau_min + 2:
        return None, 0.0

    # difference function via FFT autocorrelation (faster + smoother)
    # d(tau) = 2*(energy_cum - r(tau)) style; use classic loop on short window
    d = np.empty(tau_max + 1)
    d[0] = 0.0
    for tau in range(1, tau_max + 1):
        diff = x[: n - tau] - x[tau:]
        d[tau] = np.dot(diff, diff)

    cmnd = np.ones(tau_max + 1)
    run = 0.0
    for tau in range(1, tau_max + 1):
        run += d[tau]
        cmnd[tau] = d[tau] * tau / run if run > 1e-12 else 1.0

    # find first dip under threshold
    found = None
    tau = tau_min
    while tau < tau_max:
        if cmnd[tau] < YIN_THRESH:
            while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            found = tau
            break
        tau += 1
    if found is None:
        mi = int(np.argmin(cmnd[tau_min:tau_max])) + tau_min
        if cmnd[mi] < 0.45:
            found = mi
        else:
            return None, float(1.0 - cmnd[mi])

    # parabolic refine
    if 1 <= found < tau_max:
        s0, s1, s2 = cmnd[found - 1], cmnd[found], cmnd[found + 1]
        denom = 2 * (2 * s1 - s2 - s0)
        better = found + (s0 - s2) / denom if abs(denom) > 1e-12 else float(found)
    else:
        better = float(found)
    freq = sr / better
    conf = 1.0 - float(cmnd[found])
    if not (FMIN <= freq <= FMAX):
        return None, conf
    return freq, conf


def acf_pitch(frame: np.ndarray, sr: int = SR):
    """Normalized autocorrelation peak pitch."""
    x = frame - np.mean(frame)
    x *= np.hanning(len(x))
    # lowpass-ish: simple moving average
    k = 5
    x = np.convolve(x, np.ones(k) / k, mode="same")
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2 :]
    if corr[0] < 1e-12:
        return None, 0.0
    corr = corr / corr[0]
    tau_min = max(2, int(sr / FMAX))
    tau_max = min(len(corr) - 2, int(sr / FMIN))
    seg = corr[tau_min:tau_max]
    if len(seg) < 3:
        return None, 0.0
    # peak
    peak_i = int(np.argmax(seg)) + tau_min
    # refine: ensure local max
    if peak_i + 1 < len(corr) and peak_i > 0:
        if corr[peak_i] < corr[peak_i - 1] or corr[peak_i] < corr[peak_i + 1]:
            # search local maxima
            cands = []
            for i in range(tau_min + 1, tau_max - 1):
                if corr[i] > corr[i - 1] and corr[i] >= corr[i + 1] and corr[i] > 0.3:
                    cands.append((corr[i], i))
            if not cands:
                return None, float(corr[peak_i])
            peak_i = max(cands)[1]
    conf = float(corr[peak_i])
    if conf < 0.35:
        return None, conf
    # parabolic
    if 1 <= peak_i < len(corr) - 1:
        s0, s1, s2 = corr[peak_i - 1], corr[peak_i], corr[peak_i + 1]
        denom = 2 * (2 * s1 - s2 - s0)
        better = peak_i + (s0 - s2) / denom if abs(denom) > 1e-12 else float(peak_i)
    else:
        better = float(peak_i)
    freq = sr / better
    if not (FMIN <= freq <= FMAX):
        return None, conf
    return freq, conf


def spectral_centroid_hz(frame: np.ndarray, sr: int = SR) -> float:
    w = frame * np.hanning(len(frame))
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), 1 / sr)
    s = spec.sum()
    if s < 1e-12:
        return 0.0
    return float(np.dot(freqs, spec) / s)


def dominant_fft_peak(frame: np.ndarray, sr: int = SR):
    """Peak in magnitude spectrum in guitar fundamental range, then check harmonics."""
    n = len(frame)
    w = frame * np.hanning(n)
    # zero-pad for resolution
    nfft = 4096
    spec = np.abs(np.fft.rfft(w, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    mask = (freqs >= FMIN) & (freqs <= FMAX)
    if not np.any(mask):
        return None, 0.0
    sub = spec.copy()
    sub[~mask] = 0
    # harmonic product spectrum lightweight: multiply with downsampled
    hps = sub.copy()
    for h in range(2, 5):
        stretched = np.zeros_like(hps)
        idxs = np.arange(0, len(hps) // h) * h
        stretched[: len(idxs)] = sub[idxs]
        hps *= (stretched + 1e-12)
    peak_i = int(np.argmax(hps))
    peak_val = float(hps[peak_i])
    floor = float(np.median(hps[mask])) + 1e-12
    snr = peak_val / floor
    if snr < 5:
        return None, 0.0
    freq = float(freqs[peak_i])
    return freq, min(1.0, math.log10(snr) / 3)


def main():
    root = Path("/Users/usertrv/Kwork/projects/guitar-tabs-rihanna")
    audio = read_wav(root / "audio.wav")
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak
    lp = lowpass(audio, 1000.0)

    win = int(SR * WIN_MS / 1000)
    hop = int(SR * HOP_MS / 1000)

    rows = []
    rms_series = []
    i = 0
    while i + win <= len(audio):
        t = i / SR
        frame = lp[i : i + win]
        frame_raw = audio[i : i + win]
        rms = float(np.sqrt(np.mean(frame_raw**2)))
        rms_series.append((t, rms))
        if rms < RMS_SILENCE:
            rows.append((f"{t:.3f}", "", "", "silence", f"rms={rms:.4f}"))
            i += hop
            continue

        # try methods, pick best confidence
        cands = []
        for name, fn in (("yin", yin_pitch), ("acf", acf_pitch), ("hps", dominant_fft_peak)):
            f, c = fn(frame)
            if f is not None:
                cands.append((c, f, name))
        if not cands:
            rows.append((f"{t:.3f}", "", "", "unvoiced", f"rms={rms:.4f}"))
        else:
            cands.sort(reverse=True)
            conf, freq, method = cands[0]
            # majority among top if close
            notes = [midi_to_note(freq_to_midi(f)) for _, f, _ in cands[:3]]
            midi = freq_to_midi(freq)
            note = midi_to_note(midi)
            cents = (midi - round(midi)) * 100
            agree = notes.count(note)
            rows.append(
                (
                    f"{t:.3f}",
                    f"{freq:.2f}",
                    note,
                    f"conf={conf:.2f} method={method} agree={agree}/{min(3,len(cands))} cents={cents:+.0f}",
                    f"rms={rms:.4f}",
                )
            )
        i += hop

    out = root / "pitch.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "freq_hz", "note", "meta", "rms"])
        for r in rows:
            w.writerow(r[:4] if len(r) == 4 else r[:4])  # keep 4 cols as requested
            # actually write all useful
        f.seek(0)
        f.truncate()
        w = csv.writer(f)
        w.writerow(["time_s", "freq_hz", "note", "meta"])
        for r in rows:
            meta = r[3] if len(r) == 4 else f"{r[3]} {r[4]}"
            w.writerow([r[0], r[1], r[2], meta])

    print(f"wrote {len(rows)} rows")

    # Energy per second
    print("\n=== RMS per second ===")
    for sec in range(27):
        vals = [r for t, r in rms_series if sec <= t < sec + 1]
        if vals:
            print(f"  t={sec:02d}s  mean={np.mean(vals):.4f}  max={np.max(vals):.4f}")

    # Merge notes with median filter per 100ms buckets of voiced
    print("\n=== voiced note runs (min 40ms) ===")
    last = None
    start = None
    count = 0
    for r in rows:
        t = float(r[0])
        note = r[2]
        if not note:
            if last and count >= 4:
                print(f"  {start:.2f}–{t:.2f}  {last}  ({count} hops)")
            last = None
            count = 0
            continue
        if note != last:
            if last and count >= 4:
                print(f"  {start:.2f}–{t:.2f}  {last}  ({count} hops)")
            last = note
            start = t
            count = 1
        else:
            count += 1
    if last and count >= 4:
        print(f"  {start:.2f}–{float(rows[-1][0]):.2f}  {last}  ({count} hops)")

    # Per-second majority note
    print("\n=== majority note per second ===")
    for sec in range(27):
        notes = [r[2] for r in rows if sec <= float(r[0]) < sec + 1 and r[2]]
        if not notes:
            print(f"  t={sec:02d}s  —")
            continue
        from collections import Counter

        c = Counter(notes)
        top = c.most_common(3)
        print(f"  t={sec:02d}s  {top}")


if __name__ == "__main__":
    main()
