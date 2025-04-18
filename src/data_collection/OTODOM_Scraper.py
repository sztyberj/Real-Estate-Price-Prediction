from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# -------------------------- Konfiguracja globalna ---------------------------
BASE_URL     = "https://www.otodom.pl"
LISTING_URL  = (
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/"
    "warszawa?page="
)

# Folder pliku:  .../src/data_collection/
THIS_FILE = Path(__file__).resolve()

# ROOT_DIR = .../Real Estate Price Prediction
ROOT_DIR = THIS_FILE.parents[2]

RAW_DIR  = ROOT_DIR / "data" / "raw"
LOG_DIR  = ROOT_DIR / "logs"
TMP_DIR  = ROOT_DIR / "tmp"

for d in (RAW_DIR, LOG_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]
REQUEST_TIMEOUT = 15
FIELD_NAMES = [
    "source","price","price_per_meter","area","rooms","floor","market_type",
    "furnished","description","district","date","url","title","building_type",
    "year_built","rent","finish_status","ownership","heating","elevator",
    "ad_id","external_id",
]

# ----------------------------- Logger ---------------------------------------
LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    datefmt=DATE_FMT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"otodom_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# ----------------------------- Model danych ---------------------------------
@dataclass
class PropertyRecord:
    source: str = "otodom"
    price: str = ""
    price_per_meter: str = ""
    area: str = ""
    rooms: str = ""
    floor: str = ""
    market_type: str = ""
    furnished: str = ""
    description: str = ""
    district: str = ""
    date: str = ""
    url: str = ""
    title: str = ""
    building_type: str = ""
    year_built: str = ""
    rent: str = ""
    finish_status: str = ""
    ownership: str = ""
    heating: str = ""
    elevator: str = ""
    ad_id: str = ""
    external_id: str = ""

    def to_list(self) -> List[str]:
        return [getattr(self, f, "") for f in FIELD_NAMES]

# ----------------------------- Listing scraper ------------------------------
class ListingScraper:
    @staticmethod
    def fetch_listing(page: int) -> list[str]:
        url = f"{LISTING_URL}{page}"
        logger.info("Listing %s…", page)
        hdr = {"User-Agent": random.choice(USER_AGENTS)}
        html = requests.get(url, headers=hdr, timeout=REQUEST_TIMEOUT).text
        soup = BeautifulSoup(html, "lxml")
        links = [a["href"] for a in soup.select("a[data-cy='listing-item-link']") if a.has_attr("href")]
        full_links = [urljoin(BASE_URL, l) for l in links]
        logger.info("%s linków z strony %s", len(full_links), page)
        return full_links

# ----------------------------- Offer scraper --------------------------------
class OfferScraper:
    JSON_RE = re.compile(r"<script id=\"__NEXT_DATA__\"[^>]*>(.*?)</script>", re.S)

    @staticmethod
    def _get_characteristic(lst, key, attr="value") -> str:
        for c in lst:
            if c.get("key") == key:
                return c.get(attr) or c.get("localizedValue", "")
        return ""

    @staticmethod
    def _yes_no(lst, key) -> str:
        return "Tak" if OfferScraper._get_characteristic(lst, key) else "Nie"

    @staticmethod
    def _compose_floor(lst) -> str:
        fl = OfferScraper._get_characteristic(lst, "floor_no", "localizedValue")
        tot = OfferScraper._get_characteristic(lst, "building_floors_num")
        return f"{fl}/{tot}" if fl and tot else fl or ""

    @staticmethod
    def _extract_next_data(html: str):
        m = OfferScraper.JSON_RE.search(html)
        if not m:
            raise ValueError("No __NEXT_DATA__ JSON")
        return json.loads(m.group(1))

    @staticmethod
    def parse(url: str) -> Optional[PropertyRecord]:
        hdr = {"User-Agent": random.choice(USER_AGENTS)}
        resp = requests.get(url, headers=hdr, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = OfferScraper._extract_next_data(resp.text)
        ad   = data["props"]["pageProps"]["ad"]
        char = ad.get("characteristics", [])

        # price
        price = ""
        ad_price = ad.get("price")
        if isinstance(ad_price, dict):
            price = str(ad_price.get("value", ""))
        elif ad_price:
            price = str(ad_price)
        if not price:
            price = OfferScraper._get_characteristic(char, "price")

        # ---------- DISTRICT ----------
        district = ""

        # 1) typowa ścieżka: ad.address.district.name / code
        addr = ad.get("address")
        if addr and addr.get("district"):
            d = addr["district"]
            district = (d.get("code") or d.get("name") or "").lower()

        # 2) fallback – reverseGeocoding.locations[] gdzie "locationLevel" == "district"
        if not district:
            locs = ad.get("location", {}) \
                     .get("reverseGeocoding", {}) \
                     .get("locations", [])
            for loc in locs:
                if loc.get("locationLevel") == "district":
                    district = loc.get("name", "").lower()
                    break

        rec = PropertyRecord(
            price            = price,
            district         = district,
            price_per_meter  = OfferScraper._get_characteristic(char, "price_per_m"),
            area             = OfferScraper._get_characteristic(char, "m"),
            rooms            = OfferScraper._get_characteristic(char, "rooms_num"),
            floor            = OfferScraper._compose_floor(char),
            market_type      = "pierwotny" if ad.get("market") == "PRIMARY" else "wtórny",
            furnished        = OfferScraper._yes_no(char, "equipment_types"),
            description      = ad.get("description", ""),
            date             = ad.get("createdAt", "")[:10],
            url              = url,
            title            = ad.get("title", ""),
            building_type    = OfferScraper._get_characteristic(char, "building_type"),
            year_built       = OfferScraper._get_characteristic(char, "build_year"),
            rent             = OfferScraper._get_characteristic(char, "rent"),
            finish_status    = OfferScraper._get_characteristic(char, "construction_status"),
            ownership        = OfferScraper._get_characteristic(char, "building_ownership"),
            heating          = OfferScraper._get_characteristic(char, "heating", "localizedValue"),
            elevator         = OfferScraper._yes_no(char, "lift"),
            ad_id            = str(ad.get("id", "")),
            external_id      = ad.get("externalId", ""),
        )
        return rec

    # alias dla kompatybilności z starym kodem
    fetch_offer = parse

# ----------------------------- Data manager ---------------------------------
class DataManager:
    @staticmethod
    def save_csv(recs: list[PropertyRecord], path: Path):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(FIELD_NAMES)
            w.writerows(r.to_list() for r in recs)

# ----------------------------- Orchestrator ---------------------------------
class OtodomScraper:
    def __init__(self, start_page=1, max_pages=1, workers=8):
        self.start_page = start_page
        self.max_pages  = max_pages
        self.workers    = workers
        self.records: list[PropertyRecord] = []

    # ----------------- prywatny helper -------------------------------------
    def _scrape_buffer(self, links: list[str], page_marker: int):
        """Pobiera oferty z listy linków, odkłada do self.records
        i robi częściowy zapis CSV w <tmp/>."""
        recs: list[PropertyRecord] = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            fut_to = {executor.submit(OfferScraper.fetch_offer, u): u for u in links}
            for fut in as_completed(fut_to):
                rec = fut.result()
                if rec:
                    recs.append(rec)
                    self.records.append(rec)

        DataManager.save_partial(recs, page_marker)
        logger.info("Bufor %s stron → %s rekordów", page_marker, len(recs))

    # ----------------- PUBLIC API -----------------------------------------
    def run(self):
        listings_buffer: list[str] = []

        for p in range(self.start_page, self.max_pages + self.start_page):
            links = ListingScraper.fetch_listing(p)
            listings_buffer.extend(links)

            # co PARTIAL_SAVE_EVERY stron zapis tymczasowy
            if p % PARTIAL_SAVE_EVERY == 0:
                self._scrape_buffer(listings_buffer, p)
                listings_buffer = []

        # ostatnia paczka (jeśli #stron niepodzielne przez N)
        if listings_buffer:
            self._scrape_buffer(listings_buffer, self.max_pages + self.start_page)

        # zapis finalny
        if self.records:
            out = RAW_DIR / f"otodom_{datetime.now():%Y%m%d_%H%M%S}.csv"
            DataManager.save_csv(self.records, out)
            logger.info("Zapisano %s rekordów → %s", len(self.records), out)
        else:
            logger.warning("Brak rekordów – nic nie zapisano.")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_page", type=int, default=1)
    ap.add_argument("--max_pages", type=int, default=1)
    ap.add_argument("--workers",   type=int, default=8)
    cfg = ap.parse_args()

    OtodomScraper(cfg.start_page, cfg.max_pages, cfg.workers).run()