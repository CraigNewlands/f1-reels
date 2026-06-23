"""Circuit-specific visual themes: background colour and accent colour."""

from __future__ import annotations

# Keys are substrings matched case-insensitively against the F1 event name.
# bg    = very dark background tint (replaces pure black)
# accent = vivid colour used for dividers, title underline, highlight text
_THEMES: list[tuple[str, dict]] = [
    ("bahrain",       {"bg": "#0f0900", "accent": "#C89B3C"}),  # amber/gold
    ("saudi",         {"bg": "#00080a", "accent": "#006C35"}),  # deep green
    ("australia",     {"bg": "#000d1a", "accent": "#00843D"}),  # Australian green
    ("japan",         {"bg": "#0d0008", "accent": "#BC002D"}),  # Japanese red
    ("china",         {"bg": "#0d0008", "accent": "#DE2910"}),  # Chinese red
    ("miami",         {"bg": "#00091a", "accent": "#00B5E2"}),  # Miami blue
    ("emilia",        {"bg": "#0d0000", "accent": "#009246"}),  # Italian green
    ("monaco",        {"bg": "#00040d", "accent": "#CE1126"}),  # Monaco red
    ("canada",        {"bg": "#0d0000", "accent": "#FF0000"}),  # Canadian red
    ("spain",         {"bg": "#0d0003", "accent": "#AA151B"}),  # Spanish red
    ("austria",       {"bg": "#0d0000", "accent": "#ED2939"}),  # Austrian red
    ("britain",       {"bg": "#00050d", "accent": "#012169"}),  # Union blue
    ("british",       {"bg": "#00050d", "accent": "#012169"}),
    ("hungary",       {"bg": "#0d0008", "accent": "#436F4D"}),  # Hungarian green
    ("belgium",       {"bg": "#0d0800", "accent": "#FAE042"}),  # Belgian yellow
    ("netherlands",   {"bg": "#0d0500", "accent": "#FF6B00"}),  # Dutch orange
    ("dutch",         {"bg": "#0d0500", "accent": "#FF6B00"}),
    ("italy",         {"bg": "#00080d", "accent": "#009246"}),
    ("italian",       {"bg": "#00080d", "accent": "#009246"}),
    ("azerbaijan",    {"bg": "#00080d", "accent": "#0092BC"}),  # Azeri blue
    ("singapore",     {"bg": "#0d0000", "accent": "#EF3340"}),  # Singapore red
    ("united states", {"bg": "#00050d", "accent": "#B22234"}),  # US red
    ("mexico",        {"bg": "#00080a", "accent": "#006847"}),  # Mexican green
    ("são paulo",     {"bg": "#00080a", "accent": "#009C3B"}),  # Brazilian green
    ("brazil",        {"bg": "#00080a", "accent": "#009C3B"}),
    ("las vegas",     {"bg": "#0d0014", "accent": "#C5A028"}),  # Vegas gold
    ("qatar",         {"bg": "#0a0005", "accent": "#8D1B3D"}),  # Qatar maroon
    ("abu dhabi",     {"bg": "#00070d", "accent": "#00A4E4"}),  # Abu Dhabi teal
]

_DEFAULT = {"bg": "#080c14", "accent": "#C0C0C0"}


def get_theme(event_name: str) -> dict:
    """Return {bg, accent} for the given event name, or a neutral default."""
    lower = event_name.lower()
    for key, theme in _THEMES:
        if key in lower:
            return theme
    return _DEFAULT
