# AI Shorts Generator — Working Prototype

A full local pipeline that converts a long video into short-form vertical
clips: real ffmpeg cutting, real face-aware 9:16 reframing (OpenCV), real
burned-in captions — plus clearly marked plug-in points for the AI
transcription/analysis layer (Whisper, GPT/Claude) that this sandbox
can't call (no internet access here).

## Run it locally (needs a computer)

```bash
cd backend
pip install flask numpy opencv-python-headless   # if not already installed
python3 app.py
```

Open http://localhost:5001 — upload a video, click "Generate Shorts".

## Run it from your phone (deploy to the cloud)

This repo includes `Dockerfile` + `render.yaml` for [Render](https://render.com)
— free tier, no credit card, and it installs ffmpeg automatically.

1. Push this whole folder to a **GitHub repo** (Render deploys from GitHub).
   - Easiest way if you don't use git: create a new repo on github.com →
     "Add file → Upload files" → drag in everything from this zip.
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New +** →
   **Blueprint** → connect your GitHub repo. Render reads `render.yaml`
   automatically and sets everything up (Docker build, free plan).
3. Click **Apply** — first build takes ~5-10 min (installing ffmpeg + deps).
4. Once live, Render gives you a URL like
   `https://ai-shorts-generator-xxxx.onrender.com` — open that on your
   phone, exactly like any website.

**Free tier notes:**
- No persistent disk — uploaded videos/generated clips are stored only
  while the instance is running (fine for trying it out; download clips
  before they'd get cleared).
- The instance sleeps after ~15 min idle and takes ~30-60s to wake up on
  the next visit — normal for free tier, not a bug.
- Max upload set to 250MB in `app.py` to stay within free-tier resources;
  raise `MAX_CONTENT_LENGTH` if you upgrade to a paid plan.

## How it works (matches your spec's workflow)

1. **Upload** → `POST /api/upload` saves the file, returns a `job_id`
2. **Analyze** → `pipeline.py`:
   - `extract_audio()` — ffmpeg pulls 16kHz mono audio
   - `detect_speech_regions()` — real energy-based voice activity detection
   - `analyze_transcript()` — scores windows by vocal emphasis + pacing
     variance to pick "high-potential moments" (🔌 swap for an LLM call
     that reads the actual transcript)
3. **Cut & reframe** → `crop_to_vertical()` runs Haar-cascade face detection
   per clip and crops a 9:16 window centered on the speaker, scaled to
   1080x1920
4. **Caption** → builds an `.srt` from the detected speech timing and burns
   it in with ffmpeg's `subtitles` filter
5. **Metadata** → `generate_metadata()` fills hook/title/hashtags
   (🔌 swap for an LLM call)
6. **Dashboard** → frontend polls `/api/status/<job_id>` then renders
   cards from `/api/results/<job_id>` — score, reason, preview, download

## What's real vs. stubbed

| Stage | Status |
|---|---|
| Upload, job queue, progress polling | real |
| Audio extraction, voice-activity detection | real |
| Clip cutting, 9:16 face-aware crop, caption burn-in | real (ffmpeg + OpenCV) |
| Moment scoring | heuristic (audio energy) — works, but weaker than an LLM reading words |
| Transcript text, hook/title/hashtags | placeholder — marked `PLUG-IN POINT` in `pipeline.py`, swap in Whisper API + Claude/GPT |

## Next steps to make it production-real

1. Add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and replace the two marked
   functions in `pipeline.py` — everything downstream (cutting, cropping,
   captions, dashboard) already expects their output format, so nothing
   else changes.
2. Swap Flask dev server for gunicorn + a real job queue (Celery/RQ) once
   videos get long — the current threading approach works but won't scale
   past a few concurrent jobs.
3. Add the "Future Features" from your spec (auth, history, publishing) as
   a separate layer on top of this pipeline.
