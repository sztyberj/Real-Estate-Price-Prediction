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
    'market_type', 'furnished', 'description',
    'district', 'date', 'url', 'title'
]

# CSS Selectors for various fields based on provided selectors
SELECTORS = {
    'floor': "body > div:nth-child(1) > div:nth-child(1) > main:nth-child(4) > div:nth-child(4) > div:nth-child(1) > div:nth-child(2) > div:nth-child(3) > div:nth-child(3) > p:nth-child(2)",
    'market_type': "div:nth-child(9) p:nth-child(2)",
    'furnished': "div:nth-child(7) p:nth-child(2)",
    'description': "div[class='css-tn073k e2qsm8l0'] ul:nth-child(1) li:nth-child(1)",
    'date': ".eddsrqr5.css-xydenf",
}

records_otodom = []

def setup_driver():
    """Set up Selenium WebDriver with necessary options."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    # Add more diverse user-agents to avoid detection
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    
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

def extract_property_details_with_selectors(driver, record_dict):
    """Extract property details using the provided CSS selectors."""
    # Extract floor
    record_dict['floor'] = extract_text_safe(driver, SELECTORS['floor'])
    
    # Extract market type
    record_dict['market_type'] = extract_text_safe(driver, SELECTORS['market_type'])
    
    # Extract furnished status
    furnished_text = extract_text_safe(driver, SELECTORS['furnished'])
    if furnished_text:
        record_dict['furnished'] = "Tak" if furnished_text.lower() not in ["nie", "brak"] else "Nie"
    
    # Extract description
    record_dict['description'] = extract_text_safe(driver, SELECTORS['description'])
    
    # Extract date
    date_text = extract_text_safe(driver, SELECTORS['date'])
    if date_text:
        date_match = re.search(r'(\d{2})[./](\d{2})[./](\d{4})', date_text)
        if date_match:
            day, month, year = date_match.groups()
            record_dict['date'] = f"{year}-{month}-{day}"

def extract_property_details_fallback(driver, record_dict):
    """Fallback method to extract property details from the listing page."""
    # Target both types of property detail containers
    detail_selectors = [
        "div.css-1xw0jqp.eows69w1", 
        "div.css-4ct4xl",
        "div.css-1ww6yd9.eows59w1"  # Keep original selector for backward compatibility
    ]
    
    for selector in detail_selectors:
        try:
            detail_rows = driver.find_elements(By.CSS_SELECTOR, selector)
            for row in detail_rows:
                try:
                    # Try to extract label and value using explicit p elements
                    try:
                        label_elements = row.find_elements(By.CSS_SELECTOR, "p.eows69w2.css-1airkmu")
                        if len(label_elements) >= 2:
                            label = label_elements[0].text.strip()
                            value = label_elements[1].text.strip()
                        else:
                            # Fall back to extracting from the whole row text
                            row_text = row.text.strip()
                            if ":" in row_text:
                                parts = row_text.split(":", 1)
                                label = parts[0].strip() + ":"
                                value = parts[1].strip()
                            else:
                                continue
                    except Exception:
                        # Fall back to extracting from the whole row text
                        row_text = row.text.strip()
                        if ":" in row_text:
                            parts = row_text.split(":", 1)
                            label = parts[0].strip() + ":"
                            value = parts[1].strip()
                        else:
                            continue
                    
                    # Skip rows with no information
                    if value.lower() in ["brak informacji", "brak"]:
                        continue
                        
                    # Map labels to field names
                    if "Piętro:" in label:
                        record_dict['floor'] = value
                    elif "Rynek:" in label:
                        record_dict['market_type'] = value
                    elif "Meble:" in label:
                        record_dict['furnished'] = "Tak" if value.lower() not in ["nie", "brak"] else "Nie"
                    
                except Exception as inner_e:
                    logging.warning(f"Error parsing detail row: {inner_e}")
        except Exception as e:
            logging.warning(f"Could not extract property details with selector {selector}: {e}")

def scrap_otodom(driver, url):
    """Scrape details from an Otodom listing using Selenium."""
    if 'olx.pl' in url:
        logging.info(f"Skipping OLX URL: {url}")
        return
    
    try:
        driver.get(url)
        # Wait for page to load fully with randomized delay
        time.sleep(random.uniform(3, 6))
        
        # Initialize record dictionary
        record_dict = {field: "" for field in field_names}
        record_dict['source'] = 'otodom'
        record_dict['url'] = url
        
        # Extract title
        record_dict['title'] = extract_text_safe(driver, "h1[data-cy='adPageAdTitle']")
        
        # Extract price
        record_dict['price'] = extract_text_safe(driver, "strong[aria-label='Cena']")
        
        # Extract price per meter
        price_per_meter_selectors = ["div.css-z3xj2a.e1k1vyr25", "div[data-testid='price-per-m']"]
        for selector in price_per_meter_selectors:
            price_per_meter = extract_text_safe(driver, selector)
            if price_per_meter:
                record_dict['price_per_meter'] = price_per_meter
                break
        
        # Extract area
        try:
            area_elements = driver.find_elements(By.CSS_SELECTOR, "button.eezlw8k1.css-1nk40gi, div[data-testid='ad-features-item']")
            for element in area_elements:
                if "m²" in element.text:
                    area_text = element.text
                    area_match = re.search(r'(\d+[.,]?\d*)\s*m²', area_text)
                    if area_match:
                        record_dict['area'] = area_match.group(1)
                    break
        except Exception as e:
            logging.warning(f"Could not extract area: {e}")
        
        # Extract rooms
        try:
            room_elements = driver.find_elements(By.CSS_SELECTOR, "button.eezlw8k1.css-1nk40gi, div[data-testid='ad-features-item']")
            for element in room_elements:
                if "pokój" in element.text or "pokoje" in element.text or "pokoi" in element.text:
                    rooms_text = element.text
                    rooms_match = re.search(r'(\d+)', rooms_text)
                    if rooms_match:
                        record_dict['rooms'] = rooms_match.group(1)
                    break
        except Exception as e:
            logging.warning(f"Could not extract rooms: {e}")
        
        # Extract district
        try:
            district_elements = driver.find_elements(By.CSS_SELECTOR, "a.css-1jjm9oe.e42rcgs1, span[data-testid='location-name']")
            if district_elements:
                district_text = district_elements[0].text.strip()
                if "Warszawa," in district_text:
                    district_text = district_text.replace("Warszawa,", "").strip()
                record_dict['district'] = district_text
        except Exception as e:
            logging.warning(f"Could not extract district: {e}")
        
        # Try to extract property details using the provided CSS selectors
        try:
            extract_property_details_with_selectors(driver, record_dict)
        except Exception as e:
            logging.warning(f"Failed to extract details with provided selectors: {e}")
            # Fall back to the original method
            extract_property_details_fallback(driver, record_dict)
        
        # Append record to list
        records_otodom.append([record_dict[field] for field in field_names])
        logging.info(f"Scraped: {url}")
    
    except Exception as e:
        logging.error(f"Error scraping URL {url}: {e}")

def scrap_otodom_for_urls(driver, url, max_retries=3):
    """Scrape listing page for apartment URLs with retry mechanism."""
    for attempt in range(max_retries):
        try:
            driver.get(url)
            time.sleep(random.uniform(3, 5))
            
            # Extract listing URLs with multiple selectors for robustness
            card_links = []
            for selector in ["a[data-cy='listing-item-link']", "a[data-testid='listing-item-link']"]:
                links = driver.find_elements(By.CSS_SELECTOR, selector)
                if links:
                    card_links.extend(links)
                    break
            
            urls = list(set(link.get_attribute('href') for link in card_links if link.get_attribute('href')))
            logging.info(f"Found {len(urls)} listings on page")
            
            for link in urls:
                scrap_otodom(driver, link)
                # More variable delay to avoid detection
                time.sleep(random.uniform(2, 7))
            
            # If we get here, it means success
            return
        
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed for page {url}: {e}. Retrying...")
                time.sleep(random.uniform(10, 20))  # Longer delay before retry
            else:
                logging.error(f"Failed to extract URLs from page {url} after {max_retries} attempts: {e}")

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
    max_pages = 1  # Adjust as needed
    start_page = 1
    
    driver = setup_driver()
    logging.info("Starting Otodom scraper with Selenium...")
    
    try:
        for count in range(start_page, max_pages + 1):
            current_url = f"{otodom_url}{count}"
            logging.info(f"Processing page: {current_url}")
            
            scrap_otodom_for_urls(driver, current_url)
            
            # Save intermediate results every 5 pages
            if count % 5 == 0 or count == max_pages:
                save_to_file(f"temp_otodom_selenium_{count}.csv", records_otodom)
            
            # Variable delay between pages
            time.sleep(random.uniform(5, 10))
        
        # Save final results
        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
        save_to_file(f'raw_otodom_selenium_{timestamp}.csv', records_otodom)
        logging.info("Scraping completed successfully")
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()