import fastf1

from f1reels.config import CACHE_DIR

fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_session(year: int, round_name: str, session_type: str = "Q") -> fastf1.core.Session:
    """Load a FastF1 session with full telemetry. Results are cached locally."""
    session = fastf1.get_session(year, round_name, session_type)
    session.load()
    return session
