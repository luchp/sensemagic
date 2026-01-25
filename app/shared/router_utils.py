"""
Shared utilities for router management.
Used by discover.py, wordpress_sync.py, and test scripts.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any


# Base paths
APP_DIR = Path(__file__).parent.parent
ROUTER_DIR = APP_DIR / "routers"


def format_router_title(prefix: str) -> str:
    """
    Format router prefix into readable title.

    Args:
        prefix: Router prefix (e.g., "app_rectifier")

    Returns:
        Formatted title (e.g., "Rectifier")

    Example:
        >>> format_router_title("app_rectifier")
        'Rectifier'
        >>> format_router_title("app_power_calculator")
        'Power Calculator'
    """
    return prefix.replace("app_", "").replace("_", " ").title()


def get_router_file_path(prefix: str) -> Path:
    """
    Get the file path for a router by its prefix.

    Args:
        prefix: Router prefix (e.g., "app_rectifier")

    Returns:
        Path to the router file

    Example:
        >>> get_router_file_path("app_rectifier")
        Path('app/routers/app_rectifier.py')
    """
    return ROUTER_DIR / f"{prefix}.py"


def get_router_file_mtime(prefix: str) -> float:
    """
    Get the modification time of a router file.

    Args:
        prefix: Router prefix (e.g., "app_rectifier")

    Returns:
        Modification time (Unix timestamp), or 0 if file doesn't exist
    """
    router_file = get_router_file_path(prefix)
    if router_file.exists():
        return router_file.stat().st_mtime
    return 0


def sort_routers_by_date(routers: Dict[str, Any], reverse: bool = True) -> List[Tuple[str, Any, float]]:
    """
    Sort routers by their file modification date.

    Args:
        routers: Dictionary mapping router prefix to router object
        reverse: If True, sort newest first (descending). If False, sort oldest first.

    Returns:
        List of tuples: (prefix, router_object, mtime)
        Sorted by modification time according to reverse parameter

    Example:
        >>> routers = {"app_new": router1, "app_old": router2}
        >>> sorted_routers = sort_routers_by_date(routers)
        >>> # Returns: [("app_new", router1, 1704326400), ("app_old", router2, 1704240000)]
    """
    router_files_with_mtime = []

    for prefix, router in routers.items():
        mtime = get_router_file_mtime(prefix)
        router_files_with_mtime.append((prefix, router, mtime))

    # Sort by mtime
    router_files_with_mtime.sort(key=lambda x: x[2], reverse=reverse)

    return router_files_with_mtime


def discover_routers() -> Dict[str, Any]:
    """
    Discover all available routers using the Discover class.

    This is a convenience function for test scripts and utilities
    that need to get routers without creating a full Discover instance.

    Returns:
        Dictionary mapping router prefix to router object

    Note:
        This function imports discover.Discover to avoid circular imports.
        It's meant for use in test scripts, not production code.
    """
    # Lazy import to avoid circular dependency
    import sys
    from pathlib import Path

    # Add parent directory to path if needed
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Now import Discover
    from discover import Discover  # noqa: E402

    dis = Discover()
    dis.get_routers()
    return dis.routers


def get_router_url(domain: str, prefix: str, standalone: bool = True) -> str:
    """
    Generate a URL for a router.

    Args:
        domain: Domain name (e.g., "sensemagic.nl")
        prefix: Router prefix (e.g., "app_rectifier")
        standalone: If True, add ?standalone=true parameter

    Returns:
        Full URL to the router

    Example:
        >>> get_router_url("sensemagic.nl", "app_rectifier")
        'https://sensemagic.nl/app_rectifier/?standalone=true'
        >>> get_router_url("sensemagic.nl", "app_rectifier", standalone=False)
        'https://sensemagic.nl/app_rectifier/'
    """
    url = f"https://{domain}/{prefix}/"
    if standalone:
        url += "?standalone=true"
    return url


def separate_articles_from_apps(routers: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Separate app_articles router from other application routers.

    Args:
        routers: Dictionary of all routers

    Returns:
        Tuple of (articles_router, other_routers_dict)
        articles_router may be None if app_articles doesn't exist

    Example:
        >>> all_routers = {"app_articles": r1, "app_rectifier": r2, "app_calc": r3}
        >>> articles, apps = separate_articles_from_apps(all_routers)
        >>> articles  # r1
        >>> apps  # {"app_rectifier": r2, "app_calc": r3}
    """
    articles_router = routers.get('app_articles')
    other_routers = {k: v for k, v in routers.items() if k != 'app_articles'}
    return articles_router, other_routers

