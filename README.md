# f1-reels

Automated pipeline for generating F1 social media reels (TikTok / Instagram) from session telemetry data.

## Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/f1-reels.git
cd f1-reels
pip install -e .

# Install ffmpeg (required for video export)
brew install ffmpeg   # macOS
# apt install ffmpeg  # Ubuntu
```

## Usage

```bash
f1reels --year 2025 --round Monaco --viz qualifying-map
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | required | Season year |
| `--round` | required | Round name, e.g. `Monaco`, `Bahrain` |
| `--session` | `Q` | Session type: `Q`, `R`, `FP1`, `FP2`, `FP3` |
| `--viz` | `qualifying-map` | Visualization type |
| `--fps` | `30` | Output frame rate |
| `--duration` | `45` | Video duration in seconds |
| `--output-dir` | `./output` | Where to save the MP4 |

Output is saved to `output/<round>_<year>_<viz>.mp4` — 1080×1920 vertical, ready to upload.

## Visualizations

| Name | Description |
|------|-------------|
| `qualifying-map` | Animated dual playback of the two fastest qualifying laps on the track map |

## Adding a new visualization

1. Create `f1reels/visualizations/my_viz.py`
2. Subclass `Visualization`, set `name = "my-viz"`, implement `title()`, `setup_figure()`, `draw_frame()`
3. Run `f1reels --viz my-viz` — it's auto-discovered, no wiring required

## Automation

See `.github/workflows/render.yml` for the GitHub Actions workflow. Trigger it manually via the Actions tab after each qualifying session, or extend it to run on a schedule.

## Configuration

Copy `.env.example` to `.env` and set any overrides:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `F1_CACHE_DIR` | `~/.fastf1_cache` | FastF1 telemetry cache location |
| `F1_OUTPUT_DIR` | `./output` | Default output directory |
| `F1_FPS` | `30` | Default frame rate |
| `F1_DURATION` | `45` | Default video duration (seconds) |
