TEAM_COLORS: dict[str, str] = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#358C75",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Visa Cash App RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
}

# Fallback when team name lookup fails — keeps driver→color working across season changes
DRIVER_COLORS: dict[str, str] = {
    "VER": "#3671C6",
    "NOR": "#FF8000",
    "PIA": "#FF8000",
    "LEC": "#E8002D",
    "HAM": "#E8002D",  # Ferrari 2025
    "SAI": "#64C4FF",  # Williams 2025
    "RUS": "#27F4D2",
    "ANT": "#27F4D2",  # Antonelli, Mercedes 2025
    "ALO": "#358C75",
    "STR": "#358C75",
    "GAS": "#FF87BC",
    "DOO": "#FF87BC",  # Doohan, Alpine 2025
    "ALB": "#64C4FF",
    "COL": "#64C4FF",  # Colapinto, Williams 2025
    "TSU": "#6692FF",
    "LAW": "#6692FF",
    "HUL": "#B6BABD",
    "BEA": "#B6BABD",  # Bearman, Haas 2025
    "BOT": "#52E252",
    "BOR": "#52E252",  # Bortoleto, Sauber 2025
}

_FALLBACK = "#FFFFFF"


def driver_color(abbreviation: str, team_name: str = "") -> str:
    """Return the hex color for a driver, preferring team color lookup."""
    if team_name in TEAM_COLORS:
        return TEAM_COLORS[team_name]
    return DRIVER_COLORS.get(abbreviation.upper(), _FALLBACK)
