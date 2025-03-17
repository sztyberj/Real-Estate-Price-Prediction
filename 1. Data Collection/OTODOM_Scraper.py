import logging
import csv
import time
import random
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("apartment_scraper_otodom_selenium.log"),
        logging.StreamHandler()
    ]
)

# Constants
otodom_url = 'https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/warszawa/warszawa/warszawa?page='

# Unified field names to match OLX scraper
field_names = [
    'source', 'price', 'price_per_meter', 'area', 'rooms', 'floor',
    'market_type', 'furnished', 'building_type', 'description',
    'district', 'date', 'url', 'title', 'year_built', 'security'
]

records_otodom = []

def setup_driver():
    """Set up Selenium WebDriver with necessary options."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    # Add user-agent to avoid detection
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def extract_text_safe(driver, selector, by=By.CSS_SELECTOR, wait_time=3):
    """Safely extract text from element with error handling."""
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((by, selector))
        )
        return element.text.strip()
    except (NoSuchElementException, TimeoutException):
        return ""

def extract_elements_safe(driver, selector, by=By.CSS_SELECTOR, wait_time=3):
    """Safely extract multiple elements with error handling."""
    try:
        elements = WebDriverWait(driver, wait_time).until(
            EC.presence_of_all_elements_located((by, selector))
        )
        return elements
    except (NoSuchElementException, TimeoutException):
        return []

def scrap_otodom(driver, url):
    """Scrape details from an Otodom listing using Selenium."""
    if 'olx.pl' in url:
        logging.info(f"Skipping OLX URL: {url}")
        return
    
    try:
        driver.get(url)
        # Wait for page to load fully
        time.sleep(random.uniform(3, 5))
        
        # Initialize record dictionary
        record_dict = {field: "" for field in field_names}
        record_dict['source'] = 'otodom'
        record_dict['url'] = url
        
        # Extract title
        record_dict['title'] = extract_text_safe(driver, "h1[data-cy='adPageAdTitle']")
        
        # Extract price
        record_dict['price'] = extract_text_safe(driver, "strong[aria-label='Cena']")
        
        # Extract price per meter
        record_dict['price_per_meter'] = extract_text_safe(driver, "div.css-z3xj2a.e1k1vyr25")
        
        # Extract area
        try:
            area_elements = driver.find_elements(By.CSS_SELECTOR, "button.eezlw8k1.css-1nk40gi")
            for element in area_elements:
                if "m²" in element.text:
                    area_text = element.text
                    area_match = re.search(r'(\d+,\d+|\d+)\s*m²', area_text)
                    if area_match:
                        record_dict['area'] = area_match.group(1)
                    break
        except Exception as e:
            logging.warning(f"Could not extract area: {e}")
        
        # Extract rooms
        try:
            room_elements = driver.find_elements(By.CSS_SELECTOR, "button.eezlw8k1.css-1nk40gi")
            for element in room_elements:
                if "pokój" in element.text or "pokoje" in element.text or "pokoi" in element.text:
                    rooms_text = element.text
                    rooms_match = re.search(r'(\d+)', rooms_text)
                    if rooms_match:
                        record_dict['rooms'] = rooms_match.group(1)
                    break
        except Exception as e:
            logging.warning(f"Could not extract rooms: {e}")
        
        # Extract description
        record_dict['description'] = extract_text_safe(driver, "span.css-yl75uh.e10f53ed4")
        
        # Extract district
        try:
            district_elements = driver.find_elements(By.CSS_SELECTOR, "a.css-1jjm9oe.e42rcgs1")
            if district_elements:
                district_text = district_elements[0].text.strip()
                if "Warszawa," in district_text:
                    district_text = district_text.replace("Warszawa,", "").strip()
                record_dict['district'] = district_text
        except Exception as e:
            logging.warning(f"Could not extract district: {e}")
        
        # Extract date
        try:
            date_elements = driver.find_elements(By.CSS_SELECTOR, "p.e82kd4s2.css-htq2ld")
            if len(date_elements) >= 2:
                date_text = date_elements[1].text.strip()
                # Format date to YYYY-MM-DD
                date_match = re.search(r'(\d{2})[./](\d{2})[./](\d{4})', date_text)
                if date_match:
                    day, month, year = date_match.groups()
                    record_dict['date'] = f"{year}-{month}-{day}"
                else:
                    record_dict['date'] = date_text
        except Exception as e:
            logging.warning(f"Could not extract date: {e}")
        
        # Extract property details (floor, market type, building type, etc.)
        try:
            # Parse the key-value layout similar to what we see in the image
            detail_rows = driver.find_elements(By.CSS_SELECTOR, "div.css-1ww6yd9.eows59w1")
            for row in detail_rows:
                try:
                    # Get all text in the row
                    row_text = row.text.strip()
                    
                    # Extract various property details
                    if "Piętro:" in row_text:
                        floor_value = row_text.replace("Piętro:", "").strip()
                        record_dict['floor'] = floor_value
                    elif "Rynek:" in row_text:
                        market_value = row_text.replace("Rynek:", "").strip()
                        record_dict['market_type'] = market_value
                    elif "Rodzaj zabudowy:" in row_text:
                        building_value = row_text.replace("Rodzaj zabudowy:", "").strip()
                        record_dict['building_type'] = building_value
                    elif "Rok budowy:" in row_text:
                        year_value = row_text.replace("Rok budowy:", "").strip()
                        record_dict['year_built'] = year_value
                    elif "Zabezpieczenia:" in row_text:
                        security_value = row_text.replace("Zabezpieczenia:", "").strip()
                        record_dict['security'] = security_value
                except Exception as inner_e:
                    logging.warning(f"Error parsing detail row: {inner_e}")
        except Exception as e:
            logging.warning(f"Could not extract property details: {e}")
        
        # Extract additional information (furnished, etc.)
        try:
            feature_elements = driver.find_elements(By.CSS_SELECTOR, "p.eows59w2.css-1a1rkmu")
            for element in feature_elements:
                feature_text = element.text.lower().strip()
                if "meble" in feature_text:
                    record_dict['furnished'] = "Tak"
                    break
        except Exception as e:
            logging.warning(f"Could not extract furnished status: {e}")
        
        # Append record to list
        records_otodom.append([record_dict[field] for field in field_names])
        logging.info(f"Scraped: {url}")
    
    except Exception as e:
        logging.error(f"Error scraping URL {url}: {e}")

def scrap_otodom_for_urls(driver, url):
    """Scrape listing page for apartment URLs."""
    try:
        driver.get(url)
        time.sleep(random.uniform(3, 5))
        
        # Extract listing URLs - using the correct selector for Otodom
        card_links = driver.find_elements(By.CSS_SELECTOR, "a[data-cy='listing-item-link']")
        urls = list(set(link.get_attribute('href') for link in card_links if link.get_attribute('href')))
        
        for link in urls:
            scrap_otodom(driver, link)
            time.sleep(random.uniform(2, 4))
    
    except Exception as e:
        logging.error(f"Error extracting URLs from page {url}: {e}")

def save_to_file(name, records):
    """Save scraped records to CSV file."""
    try:
        with open(name, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(field_names)
            writer.writerows(records)
        logging.info(f"Data successfully saved to {name}")
    except Exception as e:
        logging.error(f"Error saving data to {name}: {e}")

def main():
    """Main function to control the scraping process."""
    max_pages = 2  # Adjust as needed
    start_page = 1
    
    driver = setup_driver()
    logging.info("Starting Otodom scraper with Selenium...")
    
    try:
        for count in range(start_page, max_pages + 1):
            current_url = f"{otodom_url}{count}"
            logging.info(f"Processing page: {current_url}")
            
            scrap_otodom_for_urls(driver, current_url)
            
            # Save intermediate results every 5 pages
            if count % 5 == 0:
                save_to_file(f"temp_otodom_selenium_{count}.csv", records_otodom)
            
            time.sleep(random.uniform(3, 6))
        
        # Save final results
        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
        save_to_file(f'raw_otodom_selenium_{timestamp}.csv', records_otodom)
        logging.info("Scraping completed successfully")
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()