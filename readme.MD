# Real Estate Price Prediction 🏠

![GitHub last commit](https://img.shields.io/github/last-commit/sztyberj/Real-Estate-Price-Prediction?style=for-the-badge&color=blue)
![GitHub repo size](https://img.shields.io/github/repo-size/sztyberj/Real-Estate-Price-Prediction?style=for-the-badge&color=green)
![GitHub stars](https://img.shields.io/github/stars/sztyberj/Real-Estate-Price-Prediction?style=for-the-badge&color=yellow)
![GitHub forks](https://img.shields.io/github/forks/sztyberj/Real-Estate-Price-Prediction?style=for-the-badge&color=orange)

> A comprehensive machine learning project that predicts real estate prices based on various features. This project covers the entire data science lifecycle, from data acquisition and cleaning to model deployment via a RESTful API and an interactive web interface.

---

## 🚀 Overview

This project aims to build and deploy a robust machine learning model to accurately predict real estate prices. The end-to-end pipeline includes data gathering, exploratory data analysis (EDA), feature engineering, model training, and deployment. The model is accessible through a FastAPI-powered API, and a user-friendly interface is provided using Streamlit. The entire application is containerized with Docker for easy scalability and deployment.

---

## ✨ Features

* **Data-driven Predictions**: Utilizes a comprehensive dataset of real estate properties.
* **End-to-End ML Pipeline**: Implements all stages of a data science project.
* **RESTful API**: Exposes the prediction model through a `FastAPI` endpoint.
* **Interactive UI**: A `Streamlit` web application for easy interaction with the model.
* **Containerized Application**: `Docker` setup for both the API and the web app for seamless deployment.
* **Centralized Configuration**: All parameters and settings are managed via a `config.toml` file.
* **Robust logging**: A dedicated `logger` instance is used in every script to track actions and facilitate debugging.

---

## 🛠️ Tech Stack

| Category          | Technology                                                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`                                                                                                             |
| **ML Operations** | `FastAPI`,`Pydantic`,`Uvicorn` ,`Streamlit`, `Docker`                                                                                                                                       |
| **Languages** | `Python`                                                                                                                                                               |
| **Tools** | `Jupyter Notebooks`, `Git & GitHub`, `TOML`                                                                                                                                    |

---

## 📂 Project Structure

The project is organized into several key stages, each handled by dedicated scripts and notebooks:

1.  **Data Acquisition**: Crawler to download and build the dataset.
2.  **EDA & Preprocessing**: In-depth exploratory data analysis and data cleaning notebooks.
3.  **Feature Engineering**: Creation of new features to improve model performance.
4.  **Model Training**: Training and evaluation of various regression models.
5.  **Pipeline Construction**: Building a reproducible machine learning pipeline.
6.  **Configuration**: A `config.toml` file centralized all script parameters, file path and model settings.
7.  **API Development**: Creating a `FastAPI` service to serve the model.
8.  **UI Development**: Building an interactive `Streamlit` interface.
9.  **Containerization**: `Dockerfiles` for the API and the Streamlit app.

---

## 📸 Application Preview

### Streamlit Interface
Here's a look at application, where you can interactively predict real estate prices:
![Streamlit App Screenshot](https://github.com/user-attachments/assets/0b3086a0-7da3-4928-981a-365630270cb4)

### API Documentation (FastAPI)
API is fully documented and available for testing thanks to the Swagger UI:
![FastAPI Docs Screenshot](https://github.com/user-attachments/assets/f6f3dbf9-98e7-4b3c-8527-08bf6a3c7863)

### Few plots from EDA
Box Plot of Real Estate Prices Across Districts
![BoxPlot](https://github.com/user-attachments/assets/b1782cb5-bc85-4598-bccf-535d63775ce2)

Correlation Matrix Heatmap
![Heatmap](https://github.com/user-attachments/assets/86ef8636-e709-4db5-975b-de8569e9a50e)

Relationship Between Property Area and Price
![AreaPrice](https://github.com/user-attachments/assets/b86ed4b6-2b05-499e-a40e-31085a31b41b)

### Model training
Pipeline in action
![Pipeline](https://github.com/user-attachments/assets/657548f8-4b2d-4da2-82e2-459dd7306b96)
---

## 🏁 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Docker installed on your machine.
* [Docker](https://www.docker.com/get-started)

### Installation & Launch

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/sztyberj/Real-Estate-Price-Prediction.git](https://github.com/sztyberj/Real-Estate-Price-Prediction.git)
    cd Real-Estate-Price-Prediction
    ```

2.  **Build and run the Docker containers:**
    ```sh
    docker-compose up --build
    ```

3.  **Access the applications:**
    * **FastAPI (API)**: Open your browser and go to `http://localhost:8000`
    * **Streamlit (App)**: Open your browser and go to `http://localhost:8501`

---

## 📈 Results

The final model achieves a high level of accuracy in predicting real estate prices. Here's a snapshot of the model's performance:

* **R-squared ($R^2$)**: [0.9897]
* **Root Mean Squared Error (RMSE)**: [0.0108]

## 🆕 Update 
`Date: 10.08.2025`
* **Data Leakage Elimination:** The main pipeline was rebuilt to ensure all data processing and feature engineering operations occur after the train-test split. Removed logic that generated features from the target variable **(price)** before splitting the data, ensuring a reliable model evaluation.
* **Transformer Refactoring (DataProcessor, FeatureEngineer):** 
    * Ensured full compatibility with the scikit-learn API through correct inheritance and implementation of the fit method to return self.
* **Feature Importance Analysis:** Conducted a feature importance analysis for the tuned XGBoost model, identifying the most influential features that drive price predictions and improving model interpretability.
* **GridSearchCV:** Implemented a systematic hyperparameter search using GridSearchCV for the top-performing models (Random Forest and XGBoost), which significantly improved their predictive accuracy based on cross-validated R2 scores
* **Configuration and Dependency Improvements:**
    * Fixed critical syntax errors in the config.toml file that prevented parameters from being loaded correctly.
    * Added matplotlib and seaborn as project dependencies in Poetry to enable data visualization.
