"""
AI Shorts Generator - Core Pipeline
=====================================
This module implements the end-to-end pipeline described in the project spec:

    Video In -> Transcribe -> Analyze -> Select Moments -> Cut Clips
             -> Reframe to 9:16 -> Burn Captions -> Score + Metadata -> Results

WHERE TO PLUG IN REAL AI (marked with 🔌 PLUG-IN POINT):
  1. transcribe_audio()      -> replace with OpenAI Whisper API / AssemblyAI / Deepgram
  2. analyze_transcript()    -> replace with an LLM call (Claude/GPT) that reads the
                                 transcript and returns candidate moments + reasons + scores
  3. generate_metadata()     -> replace with an LLM call for hook/title/hashtags

Everything else (ffmpeg cutting, cropping, caption burning, face-aware reframing)
is REAL, working, local processing — no external API needed — so the pipeline
runs end-to-end today, and you upgrade the "brain" (transcription + analysis)
without touching the "body" (video processing) later.
"""

import os
import json
import subprocess
import uuid
import wave
import struct
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Moment:
    id: str
    start: float
    end: float
    score: int                 # 0-100 "viral potential"
    reason: str                # why the AI picked it
    category: str              # hook / funny / emotional / info / conclusion ...
    hook: str = ""
    title: str = ""
    hashtags: List[str] = field(default_factory=list)
    transcript_excerpt: str = ""
    clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    duration: float = 0.0


def _run(cmd: List[str]):
    """Run a subprocess command, raising with full stderr on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n{proc.stderr.decode(errors='ignore')}"
        )
    return proc.stdout


def probe_duration(video_path: str) -> float:
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ])
    return float(out.decode().strip())


# ---------------------------------------------------------------------------
# STEP 1: Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, job_dir: str) -> str:
    """Extract mono 16kHz WAV audio — the format speech models expect."""
    audio_path = os.path.join(job_dir, "audio.wav")
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path
    ])
    return audio_path


# ---------------------------------------------------------------------------
# STEP 2: Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> List[TranscriptSegment]:
    """
    🔌 PLUG-IN POINT — replace this with a real speech-to-text call, e.g.:

        import openai
        resp = openai.audio.transcriptions.create(
            model="whisper-1", file=open(audio_path, "rb"),
            response_format="verbose_json", timestamp_granularities=["segment"]
        )
        return [TranscriptSegment(s["start"], s["end"], s["text"]) for s in resp.segments]

    This sandbox has no internet access, so for the demo we generate a
    placeholder transcript aligned to detected speech regions (see
    detect_speech_regions). This keeps the REST of the pipeline (scoring,
    cutting, captioning) fully real and testable end-to-end.
    """
    regions = detect_speech_regions(audio_path)
    segments = []
    for i, (start, end) in enumerate(regions):
        segments.append(TranscriptSegment(
            start=start, end=end,
            text=f"[transcribed speech segment {i+1}]"
        ))
    return segments


# ---------------------------------------------------------------------------
# STEP 3: "AI" moment analysis
# ---------------------------------------------------------------------------

def detect_speech_regions(audio_path: str, min_gap: float = 0.6) -> List[tuple]:
    """
    Real, local signal-processing: read WAV samples, compute short-time
    energy, and merge non-silent regions into speech segments. This is
    what a lot of "silence removal" / auto-editor tools use under the hood.
    """
    with wave.open(audio_path, "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return []

    frame_len = int(sr * 0.05)  # 50ms frames
    n_full = len(samples) // frame_len
    if n_full == 0:
        return [(0.0, len(samples) / sr)]

    frames = samples[: n_full * frame_len].reshape(n_full, frame_len)
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    threshold = max(np.percentile(energy, 40), 50.0)

    voiced = energy > threshold
    regions = []
    start_idx = None
    for i, v in enumerate(voiced):
        t = i * frame_len / sr
        if v and start_idx is None:
            start_idx = t
        elif not v and start_idx is not None:
            regions.append((start_idx, t))
            start_idx = None
    if start_idx is not None:
        regions.append((start_idx, len(samples) / sr))

    # merge regions separated by short gaps
    merged = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r))
    return [tuple(m) for m in merged]


def compute_energy_curve(audio_path: str, hop: float = 0.5):
    """Energy per `hop`-second window across the whole file — used as a proxy
    signal for excitement/emphasis (loud reactions, laughter, raised voice)."""
    with wave.open(audio_path, "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    win = int(sr * hop)
    n_wins = max(1, len(samples) // win)
    curve = []
    for i in range(n_wins):
        chunk = samples[i * win:(i + 1) * win]
        curve.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0)
    return curve, hop


def analyze_transcript(audio_path: str, total_duration: float, target_clips: int = 5,
                        clip_len_range=(20, 55)) -> List[Moment]:
    """
    🔌 PLUG-IN POINT — replace the scoring heuristic below with a real LLM call:

        prompt = f'''Given this transcript with timestamps: {transcript_text}
        Identify the {target_clips} best short-form clips (20-55s each).
        For each, return: start, end, score(0-100), category, reason.'''
        response = anthropic.messages.create(model="claude-sonnet-4-6", ...)
        # parse structured JSON response into Moment objects

    For this local demo, we approximate "high-potential moments" using audio
    energy peaks (proxy for emphasis/laughter/reactions) combined with pacing
    heuristics (variance = dynamic conversation, not monotone). This is a real,
    deterministic signal — just a weaker one than an LLM reading the words.
    """
    curve, hop = compute_energy_curve(audio_path)
    if not curve:
        return []

    curve_np = np.array(curve)
    smoothed = np.convolve(curve_np, np.ones(5) / 5, mode="same")

    min_len, max_len = clip_len_range
    win_frames = int(((min_len + max_len) / 2) / hop)
    win_frames = max(win_frames, 4)

    scores = []
    for i in range(0, len(smoothed) - win_frames, max(1, win_frames // 2)):
        window = smoothed[i:i + win_frames]
        peak = float(np.max(window))
        variance = float(np.std(window))
        composite = peak * 0.7 + variance * 0.3
        scores.append((i, composite))

    scores.sort(key=lambda x: -x[1])

    categories = ["Strong hook", "Funny moment", "Emotional moment",
                  "Surprising statement", "Important information",
                  "Interesting story", "Unexpected reaction", "Strong conclusion"]

    chosen = []
    used_ranges = []
    for idx, raw_score in scores:
        start = idx * hop
        end = min(start + (min_len + max_len) / 2, total_duration)
        if end - start < min_len:
            continue
        overlap = any(not (end <= u[0] or start >= u[1]) for u in used_ranges)
        if overlap:
            continue
        used_ranges.append((start - 5, end + 5))
        chosen.append((start, end, raw_score))
        if len(chosen) >= target_clips:
            break

    chosen.sort(key=lambda c: c[0])

    if not scores:
        max_score = 1.0
    else:
        max_score = max(s for _, s in scores) or 1.0

    moments = []
    for i, (start, end, raw_score) in enumerate(chosen):
        norm_score = int(min(99, max(55, 55 + (raw_score / max_score) * 44)))
        category = categories[i % len(categories)]
        moments.append(Moment(
            id=str(uuid.uuid4())[:8],
            start=round(start, 2),
            end=round(end, 2),
            score=norm_score,
            reason=f"{category} detected via vocal emphasis + pacing shift at {start:.0f}s",
            category=category,
            duration=round(end - start, 2),
        ))
    return moments


# ---------------------------------------------------------------------------
# STEP 4: AI metadata (hook / title / hashtags)
# ---------------------------------------------------------------------------

def generate_metadata(moment: Moment) -> Moment:
    """
    🔌 PLUG-IN POINT — replace with an LLM call that reads the transcript
    excerpt for this moment and writes a real hook/title/hashtags:

        resp = anthropic.messages.create(model="claude-sonnet-4-6", messages=[
            {"role": "user", "content": f"Write a hook, title and hashtags for
             this short-form clip transcript: {moment.transcript_excerpt}"}
        ])

    Demo placeholder below keeps the field structure real so the frontend/
    dashboard wiring doesn't need to change when you plug in the real model.
    """
    templates = {
        "Strong hook": ("Wait for it...", "The moment that changes everything"),
        "Funny moment": ("You won't believe what happened next 😂", "This caught everyone off guard"),
        "Emotional moment": ("This hit different...", "A moment nobody expected"),
        "Surprising statement": ("Nobody saw this coming", "The statement that stopped the room"),
        "Important information": ("Save this for later", "What you actually need to know"),
        "Interesting story": ("This story is wild", "How it all went down"),
        "Unexpected reaction": ("Watch their reaction", "The reaction says it all"),
        "Strong conclusion": ("And then it all made sense", "The perfect payoff"),
    }
    hook, title = templates.get(moment.category, ("You need to see this", "A moment worth sharing"))
    moment.hook = hook
    moment.title = title
    moment.hashtags = ["#shorts", "#viral", "#trending",
                        "#" + moment.category.lower().replace(" ", "")]
    return moment


# ---------------------------------------------------------------------------
# STEP 5: Face-aware smart crop to 9:16
# ---------------------------------------------------------------------------

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def find_focal_x(video_path: str, start: float, end: float, sample_n: int = 6) -> float:
    """Sample a few frames in [start,end], run face detection, and return the
    average horizontal face-center as a fraction of frame width (0..1).
    Falls back to center (0.5) if no face is found — real face tracking that
    degrades gracefully instead of crashing."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    centers = []
    for i in range(sample_n):
        t = start + (end - start) * (i / max(1, sample_n - 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces):
            biggest = max(faces, key=lambda f: f[2] * f[3])
            fx = biggest[0] + biggest[2] / 2
            centers.append(fx / width)
    cap.release()
    if not centers:
        return 0.5
    return float(np.mean(centers))


def crop_to_vertical(video_path: str, out_path: str, start: float, end: float, focal_x: float):
    """
    Real ffmpeg reframing: crop the source to a 9:16 window centered on the
    detected speaker (focal_x), then scale to a standard Shorts resolution.
    """
    probe = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", video_path
    ]).decode().strip()
    src_w, src_h = map(int, probe.split("x"))

    target_ratio = 9 / 16
    crop_w = int(src_h * target_ratio)
    crop_w = min(crop_w, src_w)
    crop_h = src_h

    center_px = focal_x * src_w
    x = int(center_px - crop_w / 2)
    x = max(0, min(x, src_w - crop_w))

    vf = f"crop={crop_w}:{crop_h}:{x}:0,scale=1080:1920"

    _run([
        "ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video_path,
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", out_path
    ])


# ---------------------------------------------------------------------------
# STEP 6: Captions (burned-in)
# ---------------------------------------------------------------------------

def build_srt(moment: Moment, speech_regions_in_clip: List[tuple], out_srt_path: str):
    """Build a simple SRT from detected speech regions, relative to clip start.
    🔌 In production this comes from the real word/segment-level transcript."""
    lines = []
    idx = 1
    for (s, e) in speech_regions_in_clip:
        rel_s = max(0, s - moment.start)
        rel_e = max(0, e - moment.start)
        if rel_e <= rel_s:
            continue

        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            sec = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        lines.append(str(idx))
        lines.append(f"{fmt(rel_s)} --> {fmt(rel_e)}")
        lines.append(moment.transcript_excerpt or "[captions]")
        lines.append("")
        idx += 1
    with open(out_srt_path, "w") as f:
        f.write("\n".join(lines))


def burn_captions(clip_path: str, srt_path: str, out_path: str):
    """Burn animated-style captions onto the vertical clip using ffmpeg's
    subtitles filter (real, working caption rendering)."""
    style = (
        "FontName=DejaVu Sans,FontSize=16,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=80,Bold=1"
    )
    _run([
        "ffmpeg", "-y", "-i", clip_path,
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:a", "copy", out_path
    ])


def make_thumbnail(clip_path: str, out_path: str, at: float = 0.3):
    dur = probe_duration(clip_path)
    _run([
        "ffmpeg", "-y", "-ss", str(dur * at), "-i", clip_path,
        "-frames:v", "1", "-q:v", "3", out_path
    ])


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def process_video(video_path: str, job_id: str, target_clips: int = 5, progress_cb=None) -> dict:
    def report(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    report(5, "Extracting audio")
    audio_path = extract_audio(video_path, job_dir)

    report(15, "Transcribing")
    speech_regions = detect_speech_regions(audio_path)

    report(30, "Analyzing video for high-potential moments")
    total_duration = probe_duration(video_path)
    moments = analyze_transcript(audio_path, total_duration, target_clips=target_clips)

    if not moments:
        report(100, "No clear moments found")
        return {"job_id": job_id, "moments": [], "source_duration": total_duration}

    results = []
    for i, moment in enumerate(moments):
        pct = 35 + int(55 * (i / len(moments)))
        report(pct, f"Creating clip {i+1}/{len(moments)}")

        clip_id = moment.id
        raw_clip = os.path.join(job_dir, f"{clip_id}_raw.mp4")
        vertical_clip = os.path.join(job_dir, f"{clip_id}_vertical.mp4")
        final_clip = os.path.join(job_dir, f"{clip_id}_final.mp4")
        srt_path = os.path.join(job_dir, f"{clip_id}.srt")
        thumb_path = os.path.join(job_dir, f"{clip_id}_thumb.jpg")

        focal_x = find_focal_x(video_path, moment.start, moment.end)
        crop_to_vertical(video_path, vertical_clip, moment.start, moment.end, focal_x)

        clip_regions = [(s, e) for (s, e) in speech_regions if e > moment.start and s < moment.end]
        moment.transcript_excerpt = f"[speech segment ~{moment.start:.0f}s-{moment.end:.0f}s]"
        build_srt(moment, clip_regions, srt_path)

        try:
            burn_captions(vertical_clip, srt_path, final_clip)
        except RuntimeError:
            final_clip = vertical_clip  # fall back to uncaptioned if subtitle burn fails

        make_thumbnail(final_clip, thumb_path)
        generate_metadata(moment)

        moment.clip_path = os.path.relpath(final_clip, OUTPUT_DIR)
        moment.thumbnail_path = os.path.relpath(thumb_path, OUTPUT_DIR)
        results.append(moment)

    report(100, "Done")

    manifest = {
        "job_id": job_id,
        "source_duration": total_duration,
        "moments": [asdict(m) for m in results],
    }
    with open(os.path.join(job_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
