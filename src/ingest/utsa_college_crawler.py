"""
UTSA Undergraduate Academic Unit Discovery

INGEST (discovery + classification):
- Discovers first-level undergraduate units
- Identifies which units contain course inventories

No course scraping yet.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set


UNDERGRAD_ROOT = "https://catalog.utsa.edu/undergraduate/"


def discover_undergraduate_units() -> List[str]:
    """
    Discover all first-level undergraduate units.
    """

    resp = requests.get(UNDERGRAD_ROOT, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    units: Set[str] = set()

    for a in soup.select("a[href]"):
        href = a["href"]

        if not href.startswith("/undergraduate/"):
            continue

        full = urljoin(UNDERGRAD_ROOT, href.split("#")[0])
        path = urlparse(full).path.rstrip("/")

        if path.count("/") == 2:
            units.add(full + "/")

    return sorted(units)


def unit_has_course_inventory(unit_url: str) -> bool:
    """
    Check whether a unit contains any course inventory links.
    """

    try:
        resp = requests.get(unit_url, timeout=30)
        resp.raise_for_status()
    except Exception:
        return False

    soup = BeautifulSoup(resp.text, "html.parser")

    return any(
        "#courseinventory" in a.get("href", "")
        for a in soup.select("a[href]")
    )


def discover_academic_units() -> List[str]:
    """
    Return only undergraduate units that actually expose course inventories.
    """

    units = discover_undergraduate_units()

    academic_units: List[str] = []

    for u in units:
        if unit_has_course_inventory(u):
            academic_units.append(u)

    return academic_units
