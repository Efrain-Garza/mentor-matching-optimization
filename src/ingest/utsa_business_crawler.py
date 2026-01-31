"""
UTSA Business Catalog Crawler (One Level Up)

INGEST stage:
- Starts from Business root
- Discovers all immediate sub-pages
- Scrapes any page that contains a course inventory
- Extracts titles and descriptions

No assumptions about department names.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set


BUSINESS_ROOT = "https://catalog.utsa.edu/undergraduate/business/"


def discover_business_subpages() -> Set[str]:
    """
    Discover all immediate sub-pages under Business.
    """

    resp = requests.get(BUSINESS_ROOT, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    subpages: Set[str] = set()

    for a in soup.select("a[href]"):
        href = a["href"]

        if href.startswith("/undergraduate/business/"):
            # Normalize and strip anchors
            full = urljoin(BUSINESS_ROOT, href.split("#")[0])

            # Only keep pages directly under business/
            path = urlparse(full).path.rstrip("/")
            if path.count("/") == 3:
                subpages.add(full + "/")

    return subpages


def scrape_course_inventory(page_url: str) -> List[Dict[str, str]]:
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
            "college": "business",
            "program": program,
            "source_url": inventory_url,
            "title": title_el.get_text(strip=True),
            "description": desc_el.get_text(strip=True),
        })

    return courses


def crawl_business_catalog() -> List[Dict[str, str]]:
    """
    Crawl Business catalog starting one level up.
    """

    pages = discover_business_subpages()

    all_courses: List[Dict[str, str]] = []

    for page in sorted(pages):
        all_courses.extend(scrape_course_inventory(page))

    return all_courses
