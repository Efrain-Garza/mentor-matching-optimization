"""
UTSA Course Catalog Scraper (Carlos Alvarez College of Business)

INGEST stage:
- Scrapes multiple department course inventories
- Extracts course title and description
- Tags each record with department
- Returns structured records for persistence

No cleaning, similarity, or optimization logic belongs here.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict


def scrape_catalog(cfg) -> List[Dict[str, str]]:
    """
    Scrape UTSA course catalogs for multiple business departments.

    Parameters
    ----------
    cfg : dict
        Project configuration dictionary.

    Returns
    -------
    List[Dict[str, str]]
        Each dict contains: department, title, description
    """

    urls = cfg["data"]["utsa_catalog_urls"]
    all_courses: List[Dict[str, str]] = []

    for department, url in urls.items():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for block in soup.select(".courseblock"):
            title_el = block.select_one(".courseblocktitle")
            desc_el = block.select_one(".courseblockdesc")

            if title_el is None or desc_el is None:
                continue

            all_courses.append({
                "department": department,
                "title": title_el.get_text(strip=True),
                "description": desc_el.get_text(strip=True),
            })

    return all_courses
