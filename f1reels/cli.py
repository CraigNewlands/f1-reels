import sys
from pathlib import Path

import click

from f1reels.config import DEFAULT_DURATION, DEFAULT_FPS, OUTPUT_DIR
from f1reels.data.session import load_session
from f1reels.render.base import Renderer
from f1reels.visualizations.registry import get_visualization, list_visualizations


@click.command()
@click.option("--year", required=True, type=int, help="Season year, e.g. 2025")
@click.option("--round", "round_name", required=True, help='Round name, e.g. "Monaco"')
@click.option("--session", "session_type", default="Q", show_default=True,
              help="Session type: Q (qualifying), R (race), FP1/FP2/FP3")
@click.option("--viz", default="qualifying-map", show_default=True,
              help=f"Visualization type. Available: {', '.join(list_visualizations())}")
@click.option("--fps", default=DEFAULT_FPS, show_default=True, type=int,
              help="Output frame rate")
@click.option("--duration", default=DEFAULT_DURATION, show_default=True, type=int,
              help="Video duration in seconds")
@click.option("--output-dir", default=None, type=click.Path(),
              help=f"Output directory (default: {OUTPUT_DIR})")
def main(year, round_name, session_type, viz, fps, duration, output_dir):
    """Generate F1 social media reels from session telemetry."""
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    click.echo(f"Loading {year} {round_name} [{session_type}]...")
    try:
        session = load_session(year, round_name, session_type)
    except Exception as e:
        click.echo(f"Error loading session: {e}", err=True)
        sys.exit(1)

    click.echo(f"Preparing visualization: {viz}")
    try:
        viz_cls = get_visualization(viz)
        visualization = viz_cls(session)
    except Exception as e:
        click.echo(f"Error preparing visualization: {e}", err=True)
        sys.exit(1)

    slug = round_name.lower().replace(" ", "_")
    output_path = out_dir / f"{slug}_{year}_{viz}.mp4"

    click.echo(f"Rendering {fps}fps × {duration}s → {output_path}")
    try:
        renderer = Renderer(fps=fps, duration=duration)
        renderer.render(visualization, output_path)
    except Exception as e:
        click.echo(f"Render error: {e}", err=True)
        sys.exit(1)
