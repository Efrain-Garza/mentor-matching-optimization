"""
UTSA Academic Unit Course Crawler

INGEST stage:
- Given an academic unit root (e.g. business, sciences)
- Discovers immediate subpages
- Scrapes all course inventories
- Extracts structured course records
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set


def discover_subpages(unit_url: str) -> Set[str]:
    """
    Discover immediate subpages under an academic unit.
    """

    resp = requests.get(unit_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    subpages: Set[str] = set()

    for a in soup.select("a[href]"):
        href = a["href"]

        if not href.startswith(unit_url.replace("https://catalog.utsa.edu", "")):
            continue

        full = urljoin(unit_url, href.split("#")[0])
        path = urlparse(full).path.rstrip("/")

        # One level below the unit
        if path.count("/") == 3:
            subpages.add(full + "/")

    return subpages


def scrape_course_inventory(page_url: str, unit: str) -> List[Dict[str, str]]:
    """
    Attempt to scrape a course inventory from a page.
    """

    inventory_url = page_url.rstrip("/") + "/#courseinventory"

    resp = requests.get(inventory_url, timeout=30)
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    program = page_url.rstrip("/").split("/")[-1]

    courses: List[Dict[str, str]] = []

    for block in soup.select(".courseblock"):
        title_el = block.select_one(".courseblocktitle")
        desc_el = block.select_one(".courseblockdesc")

        if not title_el or not desc_el:
            continue

        courses.append({
            "college": unit,
            "program": program,
            "source_url": inventory_url,
            "title": title_el.get_text(strip=True),
            "description": desc_el.get_text(strip=True),
        })

    return courses


def crawl_academic_unit(unit_url: str) -> List[Dict[str, str]]:
    """
    Crawl all course inventories under a single academic unit.
    """

    unit = unit_url.rstrip("/").split("/")[-1]
    subpages = discover_subpages(unit_url)

    all_courses: List[Dict[str, str]] = []

    for page in sorted(subpages):
        all_courses.extend(scrape_course_inventory(page, unit))

    return all_courses
