"""
UTSA Undergraduate Catalog Crawler (Two-Phase)

INGEST stage:
1. Discover undergraduate program pages
2. Discover course inventory pages from each program
3. Extract course blocks

This is a controlled, catalog-aware crawl.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set


def discover_program_pages(base_url: str) -> Set[str]:
    """
    Discover undergraduate program pages.

    Returns URLs like:
    /undergraduate/business/
    /undergraduate/aicybercomputing/
    """

    resp = requests.get(base_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    programs: Set[str] = set()

    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("/undergraduate/") and href.count("/") == 2:
            programs.add(urljoin(base_url, href))

    return programs


def discover_course_inventory_links(program_url: str) -> Set[str]:
    """
    Discover course inventory links from a program page.
    """

    resp = requests.get(program_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links: Set[str] = set()

    for a in soup.select("a[href]"):
        href = a["href"]
        if "#courseinventory" in href:
            links.add(urljoin(program_url, href))

    return links


def scrape_course_inventory(url: str) -> List[Dict[str, str]]:
    """
    Scrape a single course inventory page.
    """

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    program = parts[-2] if len(parts) >= 2 else "unknown"

    courses: List[Dict[str, str]] = []

    for block in soup.select(".courseblock"):
        title_el = block.select_one(".courseblocktitle")
        desc_el = block.select_one(".courseblockdesc")

        if not title_el or not desc_el:
            continue

        courses.append({
            "program": program,
            "source_url": url,
            "title": title_el.get_text(strip=True),
            "description": desc_el.get_text(strip=True),
        })

    return courses


def crawl_undergraduate_catalog(cfg) -> List[Dict[str, str]]:
    """
    Crawl the full undergraduate catalog.
    """

    base_url = cfg["data"]["base_url"]

    programs = discover_program_pages(base_url)

    inventory_links: Set[str] = set()
    for program_url in programs:
        inventory_links |= discover_course_inventory_links(program_url)

    all_courses: List[Dict[str, str]] = []

    for url in sorted(inventory_links):
        all_courses.extend(scrape_course_inventory(url))

    return all_courses
