### Build with LLM 

from __future__ import annotations
import csv
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin
import toml

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------- progress‑bar (optional) ---------------------------------
try:
    from tqdm import tqdm          
except ImportError:                
    tqdm = None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_URL   = "https://www.otodom.pl"
LISTING_URL = (
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/"
    "warszawa?page="
)

# --- project paths -------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[2]          # …/Real Estate Price Prediction
RAW_DIR   = ROOT_DIR / "data" / "raw"
LOG_DIR   = ROOT_DIR / "src" / "data_collection" / "logs" 
TMP_DIR   = ROOT_DIR / "data" / "tmp"
for d in (RAW_DIR, LOG_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)
    
# --- load config --------------------------------------------------------
with open(ROOT_DIR / "config.toml", 'r') as f:
    config = toml.load(f)

REQUEST_TIMEOUT = config['data_scraping']['request_timeout']
MAX_RETRIES = config['data_scraping']['max_retries']       
BASE_DELAY_SEC = config['data_scraping']['base_delay_sec']
PARTIAL_SAVE_EVERY = config['data_scraping']['partial_save_energy']
START_PAGE = config['data_scraping']['start_page']
MAX_PAGES = config['data_scraping']['max_pages']
WORKERS = config['data_scraping']['workers']
     


# --- network params --------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]

# --- logger -----------------------------------------------------------------
LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    datefmt=DATE_FMT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"otodom_{datetime.now():%Y%m%d}.log",
                            encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UTILS – GET (retry + exponential back‑off + delay)
# ---------------------------------------------------------------------------
_SESSION = requests.Session()
retry_strategy = Retry(
    total=MAX_RETRIES,
    status_forcelist=[403, 429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    backoff_factor=BASE_DELAY_SEC,
    raise_on_status=False,
)
_SESSION.mount("https://", HTTPAdapter(max_retries=retry_strategy))

def request_with_retry(url: str) -> requests.Response:
    hdr = {"User-Agent": random.choice(USER_AGENTS)}
    resp = _SESSION.get(url, headers=hdr, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        logger.warning("HTTP %s → %s", resp.status_code, url)
    time.sleep(random.uniform(BASE_DELAY_SEC * 0.8, BASE_DELAY_SEC * 1.2))
    return resp

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
FIELD_NAMES = [
    "source","price","price_per_meter","area","rooms","floor","market_type",
    "furnished","description","district","date","url","title","building_type",
    "year_built","rent","finish_status","ownership","heating","elevator",
    "ad_id","external_id",
]

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

# ---------------------------------------------------------------------------
# LISTING SCRAPER
# ---------------------------------------------------------------------------
class ListingScraper:
    @staticmethod
    def fetch_listing(page: int) -> list[str]:
        url = f"{LISTING_URL}{page}"
        html = request_with_retry(url).text
        soup = BeautifulSoup(html, "lxml")
        links = [
            a["href"] for a in soup.select("a[data-cy='listing-item-link']")
            if a.has_attr("href")
        ]
        return [urljoin(BASE_URL, l) for l in links]

# ---------------------------------------------------------------------------
# OFFER SCRAPER
# ---------------------------------------------------------------------------
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
        fl  = OfferScraper._get_characteristic(lst, "floor_no", "localizedValue")
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
        try:
            resp = request_with_retry(url)
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

            # district
            district = ""
            addr = ad.get("address")
            if addr and addr.get("district"):
                d = addr["district"]
                district = (d.get("code") or d.get("name") or "").lower()
            if not district:
                locs = ad.get("location", {}) \
                         .get("reverseGeocoding", {}) \
                         .get("locations", [])
                for loc in locs:
                    if loc.get("locationLevel") == "district":
                        district = loc.get("name", "").lower()
                        break

            return PropertyRecord(
                price            = price,
                district         = district,
                price_per_meter  = OfferScraper._get_characteristic(char,"price_per_m"),
                area             = OfferScraper._get_characteristic(char,"m"),
                rooms            = OfferScraper._get_characteristic(char,"rooms_num"),
                floor            = OfferScraper._compose_floor(char),
                market_type      = "pierwotny" if ad.get("market")=="PRIMARY" else "wtórny",
                furnished        = OfferScraper._yes_no(char,"equipment_types"),
                description      = ad.get("description",""),
                date             = ad.get("createdAt","")[:10],
                url              = url,
                title            = ad.get("title",""),
                building_type    = OfferScraper._get_characteristic(char,"building_type"),
                year_built       = OfferScraper._get_characteristic(char,"build_year"),
                rent             = OfferScraper._get_characteristic(char,"rent"),
                finish_status    = OfferScraper._get_characteristic(char,"construction_status"),
                ownership        = OfferScraper._get_characteristic(char,"building_ownership"),
                heating          = OfferScraper._get_characteristic(char,"heating","localizedValue"),
                elevator         = OfferScraper._yes_no(char,"lift"),
                ad_id            = str(ad.get("id","")),
                external_id      = ad.get("externalId",""),
            )
        except Exception as exc:
            logger.warning("Offer fail → %s : %s", url, exc)
            return None

    fetch_offer = parse   # alias

# ---------------------------------------------------------------------------
# DATA MANAGER
# ---------------------------------------------------------------------------
class DataManager:
    @staticmethod
    def save_csv(recs: list[PropertyRecord], path: Path):
        with path.open("w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f, delimiter=";")
            cw.writerow(FIELD_NAMES)
            cw.writerows(r.to_list() for r in recs)

    @staticmethod
    def save_partial(recs: list[PropertyRecord], page: int):
        if not recs:
            return
        tmp = TMP_DIR / f"page_{page:04d}_{datetime.now():%H%M%S}.csv"
        DataManager.save_csv(recs, tmp)
        logger.info("Temp save → %s (%s records)", tmp.name, len(recs))

# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------
class OtodomScraper:
    def __init__(self, start_page= START_PAGE, max_pages= MAX_PAGES, workers= WORKERS):
        self.start_page = start_page
        self.max_pages  = max_pages
        self.workers    = workers
        self.records: list[PropertyRecord] = []

    def _scrape_buffer(self, links: list[str], page_marker: int):
        recs: list[PropertyRecord] = []

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = [ex.submit(OfferScraper.fetch_offer, u) for u in links]
            fut_iter = futures
            if tqdm:
                fut_iter = tqdm(as_completed(futures),
                                total=len(futures),
                                desc=f"Oferty {page_marker}",
                                unit="ad")
            for fut in fut_iter:
                rec = fut.result()
                if rec:
                    recs.append(rec)
                    self.records.append(rec)

        DataManager.save_partial(recs, page_marker)

    def run(self):
        # -------- get listing ------------------------------------
        page_range = range(self.start_page, self.start_page + self.max_pages)
        page_iter = tqdm(page_range, desc="Listing pages", unit="page") if tqdm else page_range

        buffer: list[str] = []
        for p in page_iter:
            links = ListingScraper.fetch_listing(p)
            buffer.extend(links)

            if p % PARTIAL_SAVE_EVERY == 0:
                self._scrape_buffer(buffer, p)
                buffer = []

        if buffer:
            self._scrape_buffer(buffer, self.start_page + self.max_pages - 1)

        # -------- final save ------------------------------------------
        if self.records:
            out = RAW_DIR / f"otodom_{datetime.now():%Y%m%d_%H%M%S}.csv"
            DataManager.save_csv(self.records, out)
            logger.info("Zapisano %s rekordów → %s", len(self.records), out)
        else:
            logger.warning("No records to save.")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_page", type=int, default=1)
    ap.add_argument("--max_pages", type=int, default=1)
    ap.add_argument("--workers",   type=int, default=8)
    cfg = ap.parse_args()
    
    OtodomScraper(cfg.start_page, cfg.max_pages, cfg.workers).run()
    """
    OtodomScraper().run()