import streamlit as st
import requests
import json
import pandas as pd
import toml
from pathlib import Path

with open("config.toml", 'r') as f:
    config = toml.load(f)
    config = json.loads(json.dumps(config))

host = "api"
port = 8000

#Page Config
st.set_page_config(
    page_title="Predykcja Cen Nieruchomości",
    page_icon="🏠",
    layout="centered"
)

#Title
st.title("🏠 Predykcja Cen Mieszkań")
st.write(
    "Wprowadź dane dotyczące nieruchomości, aby otrzymać przewidywaną cenę rynkową. "
    "Aplikacja komunikuje się z modelem predykcyjnym poprzez API."
)

#API Adress
API_URL = f"http://{host}:{port}/predict/"

district_options = [
    'Mokotów', 'Praga-Południe', 'Wola', 'Ursynów', 'Bielany', 'Bemowo',
    'Śródmieście', 'Targówek', 'Ochota', 'Wawer', 'Białołęka', 'Praga-Północ',
    'Ursus', 'Żoliborz', 'Włochy', 'Wilanów', 'Wesoła', 'Rembertów'
]
market_type_options = ['pierwotny', 'wtórny']

#Polish dictionary
building_type_map = {
    'Blok': 'block',
    'Kamienica': 'tenement',
    'Apartamentowiec': 'apartment',
    'Zabudowa wypełniająca': 'infill',
    'Szeregowiec': 'ribbon',
    'Dom': 'house',
    'Loft': 'loft'
}

finish_status_map = {
    'Do zamieszkania': 'ready_to_use',
    'Do remontu': 'to_renovation',
    'Do wykończenia': 'to_completion'
}

heating_options = ['miejskie', 'gazowe', 'inne', 'kotłownia', 'elektryczne', 'piece kaflowe']

ownership_map = {
    'Pełna własność': 'full_ownership',
    'Spółdzielcze własnościowe': 'limited_ownership',
    'Udział': 'share',
    'Użytkowanie wieczyste': 'usufruct'
}

#Forms
with st.form("prediction_form"):
    st.header("Wprowadź parametry mieszkania")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Powierzchnia (m²)", min_value=10.0, max_value=300.0, value=55.0, step=0.5)
        rooms = st.number_input("Liczba pokoi", min_value=1, max_value=10, value=3, step=1)
        floor = st.number_input("Piętro", min_value=0, max_value=50, value=3, step=1)
        building_max_floor = st.number_input("Liczba pięter w budynku", min_value=1, max_value=50, value=10, step=1)
        year_built = st.number_input("Rok budowy", min_value=1900, max_value=2030, value=2015, step=1)
        rent = st.number_input("Czynsz", min_value=1, max_value=30000, value=800, step=100)

    with col2:
        district = st.selectbox("Dzielnica", options=sorted(district_options))
        market_type = st.selectbox("Rynek", options=market_type_options, index=1)
        building_type_pl = st.selectbox("Typ budynku", options=list(building_type_map.keys()))
        finish_status_pl = st.selectbox("Stan wykończenia", options=list(finish_status_map.keys()))
        ownership_pl = st.selectbox("Forma własności", options=list(ownership_map.keys()))
        heating = st.selectbox("Ogrzewanie", options=heating_options)

    st.subheader("Udogodnienia")
    amenities_col1, amenities_col2, amenities_col3, amenities_col4 = st.columns(4)
    with amenities_col1:
        garage = st.checkbox("Garaż", value=True)
        elevator = st.checkbox("Winda", value=True)
        furnished = st.checkbox("Wykończone", value=True)
    
    submit_button = st.form_submit_button(label="Predykcja ceny")

if submit_button:
    if floor > building_max_floor:
        st.error(f"Błąd: Piętro {floor} nie może być wyższe niż liczba pięter w budynku {building_max_floor}.")
    else:
        data_to_predict = {
            "area": area,
            "rooms": rooms,
            "floor": floor,
            "building_max_floor": building_max_floor,
            "year_built": year_built,
            "district": district.lower(),
            "market_type": market_type,
            "furnished": furnished,
            "elevator": elevator,
            "building_type": building_type_map.get(building_type_pl),
            "finish_status": finish_status_map.get(finish_status_pl),
            "ownership": ownership_map.get(ownership_pl),
            "heating": heating,
            "garage": garage,
            "rent": rent
        }

        with st.spinner("Przetwarzanie danych i predykcja..."):
            try:
                response = requests.post(API_URL, json=[data_to_predict])
                response.raise_for_status()

                result = response.json()
                if "predictions" in result and result["predictions"]:
                    predicted_price = result["predictions"][0]
                    formatted_price = f"{predicted_price:,.0f} PLN".replace(",", " ")
                    
                    st.success(f"**Przewidywana cena nieruchomości: {formatted_price}**")
                else:
                    st.error(f"Wystąpił błąd w odpowiedzi z API: {result.get('error', 'Brak predykcji.')}")

            except requests.exceptions.RequestException as e:
                st.error(f"Błąd połączenia z API. Upewnij się, że serwer FastAPI jest uruchomiony. Szczegóły: {e}")
            except Exception as e:
                st.error(f"Wystąpił nieoczekiwany błąd: {e}")