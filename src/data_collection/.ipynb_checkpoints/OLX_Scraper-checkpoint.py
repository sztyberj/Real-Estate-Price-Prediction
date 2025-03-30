import logging
import csv
import time
import random
import re
import os
import argparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/apartment_scraper_olx_selenium.log"),
        logging.StreamHandler()
    ]
)

warsaw_url = 'https://www.olx.pl/d/nieruchomosci/mieszkania/sprzedaz/warszawa/?page='

field_names = [
    'price', 'price_per_meter', 'area', 'rooms', 'floor',
    'market_type', 'furnished', 'building_type', 'description',
    'district', 'date', 'url'
]

months_pl = {
    'stycznia': '01', 'lutego': '02', 'marca': '03',
    'kwietnia': '04', 'maja': '05', 'czerwca': '06',
    'lipca': '07', 'sierpnia': '08', 'września': '09',
    'października': '10', 'listopada': '11', 'grudnia': '12'
}

records_olx = []

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scrap_olx(driver, url):
    if 'otodom.pl' in url:
        logging.info(f"Skipping Otodom URL: {url}")
        return

    driver.get(url)
    time.sleep(random.uniform(3, 5))  # Poczekaj na dynamiczne ładowanie strony

    record_dict = {field: "" for field in field_names}
    record_dict['url'] = url

    try:
        # Price
        try:
            price = driver.find_element(By.CSS_SELECTOR, '[data-testid="ad-price-container"]').text
            record_dict['price'] = re.sub(r'\s+', ' ', price)
        except:
            logging.warning(f"No price found for {url}")

        # PPM
        try:
            ppm = driver.find_element(By.XPATH, "//p[contains(text(), 'zł/m²')]").text.replace('Cenazam²', 'Cena za m² ')
            record_dict['price_per_meter'] = ppm
        except:
            logging.warning(f"No price per meter found for {url}")

        # Description
        try:
            description = driver.find_element(By.CSS_SELECTOR, '[data-testid="ad_description"]').text
            record_dict['description'] = description.strip()
        except:
            logging.warning(f"No description found for {url}")

        # District
        try:
            district_el = driver.find_element(By.XPATH, "//p[contains(text(), 'Warszawa')]")
            district = district_el.text.replace('Warszawa,', '').strip()
            record_dict['district'] = district
        except:
            logging.warning(f"No district found for {url}")

        # Publication date
        try:
            date_text = driver.find_element(By.CSS_SELECTOR, '[data-cy="ad-posted-at"]').text.strip()
            if 'Dzisiaj' in date_text:
                parsed_date = datetime.today()
            else:
                for pl_month, num_month in months_pl.items():
                    if pl_month in date_text:
                        date_text = date_text.replace(pl_month, num_month)
                        break
                parsed_date = datetime.strptime(date_text, '%d %m %Y')
            record_dict['date'] = parsed_date.strftime('%Y-%m-%d')
        except:
            logging.warning(f"No date found or parsing error for {url}")

        # Others params
        details = driver.find_elements(By.CSS_SELECTOR, 'div[tabindex]')
        for detail in details:
            param_text = detail.text.strip()
            if 'Powierzchnia' in param_text:
                area_match = re.search(r'(\d+,\d+|\d+)\s*m²', param_text)
                if area_match:
                    record_dict['area'] = area_match.group(1)
            elif 'Liczba pokoi' in param_text:
                rooms_match = re.search(r'Liczba pokoi[:\s]*(\d+)', param_text)
                if rooms_match:
                    record_dict['rooms'] = rooms_match.group(1)
            elif 'Poziom' in param_text:
                record_dict['floor'] = param_text.split(':')[1].strip()
            elif 'Rynek' in param_text:
                record_dict['market_type'] = param_text.split(':')[1].strip()
            elif 'Umeblowane' in param_text:
                record_dict['furnished'] = param_text.split(':')[1].strip()
            elif 'Rodzaj zabudowy' in param_text:
                record_dict['building_type'] = param_text.split(':')[1].strip()

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")

    records_olx.append([record_dict[field] for field in field_names])
    logging.info(f"Scraped: {url}")

def scrap_olx_for_urls(driver, url):
    driver.get(url)
    time.sleep(random.uniform(3, 5))

    cards = driver.find_elements(By.CSS_SELECTOR, '[data-cy="l-card"] a')
    urls = list(set(card.get_attribute('href') for card in cards if card.get_attribute('href')))

    for link in urls:
        scrap_olx(driver, link)
        time.sleep(random.uniform(2, 4))

def save_to_file(name, records):
    with open(name, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(field_names)
        writer.writerows(records)
    logging.info(f"Data successfully saved to {name}")

def main(start_page, max_pages):
    driver = setup_driver()

    for count in range(start_page, start_page + max_pages):
        current_url = f"{warsaw_url}{count}"
        logging.info(f"Processing page: {current_url}")
        scrap_olx_for_urls(driver, current_url)
        if count % 5 == 0:
            save_to_file(f"tmp/tmp_olx_{count}.csv", records_olx)
        time.sleep(random.uniform(3, 6))


    FILENAME = f'raw_olx_{datetime.now().strftime("%d%m%Y%H%M%S")}.csv'
    full_path = os.path.join(RAW_PATH, FILENAME)
    save_to_file(full_path, records_olx)

    driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape OLX with Selenium")
    parser.add_argument('--start_page', type=int, default=1, help='First page to scrape')
    parser.add_argument('--max_pages', type=int, default=1, help='Number of pages to scrape')
    args = parser.parse_args()
    main(args.start_page, args.max_pages)
