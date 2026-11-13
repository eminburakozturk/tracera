"""Playwright ile README demo videosunu kaydeder ve ffmpeg ile MP4 + GIF + 3 kare uretir.

Kullanim:
    C:\\Users\\Burak\\anaconda3\\envs\\tracera\\python.exe scripts/make_demo.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
DEMO_DIR = REPO_ROOT / "results" / "demo"

WEBM_PATH = DEMO_DIR / "tracera_demo.webm"
MP4_PATH = DEMO_DIR / "tracera_demo.mp4"
GIF_PATH = DEMO_DIR / "tracera_demo.gif"
META_PATH = DEMO_DIR / "demo_meta.json"

MAX_MP4_BYTES = 8 * 1024 * 1024
MAX_GIF_BYTES = 10 * 1024 * 1024


def _resolve_ffmpeg() -> str:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not exe or not Path(exe).is_file():
        raise FileNotFoundError(f"imageio_ffmpeg bulunamadi: {exe}")
    return exe


def _wait_api_ready(url: str, timeout_s: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception as err:  # noqa: BLE001
            last_err = err
        time.sleep(0.5)
    raise TimeoutError(f"Yerel API hazir olmadi ({url}): {last_err}")


def _run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    print("[make_demo] >", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Komut hata koduyla bitti ({res.returncode}): {' '.join(cmd)}")


def _encode_mp4(ffmpeg_exe: str, ss_sec: float) -> None:
    for crf in (23, 26, 29):
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss",
            f"{ss_sec:.2f}",
            "-i",
            str(WEBM_PATH),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(MP4_PATH),
        ]
        _run_checked(cmd)
        size = MP4_PATH.stat().st_size
        print(f"[make_demo] MP4 (crf={crf}): {size / (1024 * 1024):.2f} MB")
        if size <= MAX_MP4_BYTES:
            return
    raise RuntimeError(f"MP4 8 MB sinirini asti: {MP4_PATH.stat().st_size} B")


def _render_gif(
    ffmpeg_exe: str,
    ss_sec: float,
    to_sec: float | None,
    max_colors: int,
    stats_mode: str,
    paletteuse_opts: str = "dither=none:diff_mode=rectangle",
) -> int:
    palette_path = DEMO_DIR / "_palette.png"
    vf_base = "fps=12,scale=960:-1:flags=lanczos"
    trim_args = ["-ss", f"{ss_sec:.2f}"]
    if to_sec is not None:
        trim_args += ["-t", f"{max(1.0, to_sec - ss_sec):.2f}"]

    _run_checked(
        [
            ffmpeg_exe,
            "-y",
            *trim_args,
            "-i",
            str(WEBM_PATH),
            "-vf",
            f"{vf_base},palettegen=max_colors={max_colors}:stats_mode={stats_mode}",
            str(palette_path),
        ]
    )
    _run_checked(
        [
            ffmpeg_exe,
            "-y",
            *trim_args,
            "-i",
            str(WEBM_PATH),
            "-i",
            str(palette_path),
            "-lavfi",
            f"{vf_base} [x]; [x][1:v] paletteuse={paletteuse_opts}",
            str(GIF_PATH),
        ]
    )
    if palette_path.is_file():
        palette_path.unlink()
    return GIF_PATH.stat().st_size


def _encode_gif(
    ffmpeg_exe: str,
    ss_sec: float,
    steps12_end_sec: float,
    total_sec: float,
) -> float:
    configs = [
        (None, 256, "diff", "dither=bayer:bayer_scale=3:diff_mode=rectangle"),
        (None, 128, "diff", "dither=none:diff_mode=rectangle"),
        (None, 64, "diff", "dither=none:diff_mode=rectangle"),
        (steps12_end_sec, 128, "diff", "dither=none:diff_mode=rectangle"),
        (steps12_end_sec, 96, "diff", "dither=none:diff_mode=rectangle"),
        (steps12_end_sec, 64, "diff", "dither=none:diff_mode=rectangle"),
    ]
    for to_sec, colors, mode, puse in configs:
        size = _render_gif(ffmpeg_exe, ss_sec, to_sec, colors, mode, puse)
        scope = "tam akis" if to_sec is None else "yalniz 1-2. adimlar"
        print(
            f"[make_demo] GIF ({scope}, colors={colors}): {size / (1024 * 1024):.2f} MB"
        )
        if size <= MAX_GIF_BYTES:
            return total_sec if to_sec is None else max(5.0, to_sec - ss_sec)
    raise RuntimeError(f"GIF 10 MB sinirini asti: {GIF_PATH.stat().st_size} B")


def _extract_gif_frames(ffmpeg_exe: str, duration_guess_s: float) -> None:
    # GIF suresini ffprobe olmadan guvenli 3 ornek zaman noktasindan cikar
    timestamps = [
        1.0,
        max(3.0, duration_guess_s * 0.45),
        max(5.0, duration_guess_s * 0.85),
    ]
    for idx, ts in enumerate(timestamps, start=1):
        out_png = DEMO_DIR / f"frame_{idx:02d}.png"
        _run_checked(
            [
                ffmpeg_exe,
                "-y",
                "-ss",
                f"{ts:.2f}",
                "-i",
                str(GIF_PATH),
                "-frames:v",
                "1",
                str(out_png),
            ]
        )


def main() -> int:
    encode_only = "--encode-only" in sys.argv[1:]
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = _resolve_ffmpeg()
    print(f"[make_demo] ffmpeg: {ffmpeg_exe}")
    _run_checked([ffmpeg_exe, "-version"])

    if not encode_only:
        # 1) Yerel API'yi gecici olarak baslat (bulma / OpenVINO + LayerCAM)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        uvicorn_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ],
            cwd=str(REPO_ROOT),
            env=env,
        )
        try:
            _wait_api_ready("http://127.0.0.1:8765/health", timeout_s=60.0)
            print("[make_demo] Yerel API hazir, Playwright demo kaydi basliyor...")
            npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
            _run_checked([npm_cmd, "run", "demo:record"], cwd=WEB_DIR)
        finally:
            uvicorn_proc.terminate()
            try:
                uvicorn_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                uvicorn_proc.kill()

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    ss_sec = float(meta.get("firstPaintedSec", 0.5)) + 0.15
    steps12_end_sec = float(meta.get("steps12EndSec", 22.0))
    total_sec = max(8.0, float(meta.get("totalRecordedSec", 35.0)) - ss_sec)

    _encode_mp4(ffmpeg_exe, ss_sec)
    gif_duration_s = _encode_gif(ffmpeg_exe, ss_sec, steps12_end_sec, total_sec)
    _extract_gif_frames(ffmpeg_exe, gif_duration_s)

    if META_PATH.is_file():
        META_PATH.unlink()

    print(
        f"[make_demo] Tamamlandi:\n"
        f"  WEBM: {WEBM_PATH} ({WEBM_PATH.stat().st_size / (1024 * 1024):.2f} MB)\n"
        f"  MP4 : {MP4_PATH} ({MP4_PATH.stat().st_size / (1024 * 1024):.2f} MB)\n"
        f"  GIF : {GIF_PATH} ({GIF_PATH.stat().st_size / (1024 * 1024):.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
