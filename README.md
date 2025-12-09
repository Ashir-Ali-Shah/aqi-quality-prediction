# 🌫️ Urban Air Quality Sentinel

    
**A Full-Stack AI System for Real-time Smog Detection, Forecasting, and Health Assistance.**

**Air Quality Sentinel** is a production-ready application designed to monitor and predict PM2.5 levels in major Pakistani cities. It combines real-time data ingestion, a highly optimized Random Forest forecasting engine, and a RAG (Retrieval-Augmented Generation) chatbot to provide actionable health advice.

## 📊 Quantitative Model Performance

The core of this system is a **Random Forest Regressor (200 estimators)** optimized for low-latency inference. Below are real-world inference metrics captured during system validation.

### 1\. Inference Speed & Efficiency

The model utilizes `joblib` with a threading backend to parallelize predictions, ensuring real-time responsiveness for the user dashboard.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Inference Latency** | **\~0.1s** | Ultra-low latency prediction time per request. |
| **Estimators** | **200** | Number of decision trees in the ensemble model. |
| **Concurrency** | **2 Workers** | Parallel processing enabled for scalability. |

### 2\. Prediction Case Studies (Real Logs)

The model demonstrates high sensitivity to meteorological changes (Temperature vs. Humidity correlations).

**Case A: Moderate Heat, High Humidity**
*Input:* `PM10: 85`, `NO2: 52`, `CO: 720`, `Temp: 28°C`, `Humidity: 65%`

> **Result:** The model correctly identifies the compounding effect of humidity on particulate matter.
>
>   * **Predicted PM2.5:** `64.01 μg/m³`
>   * **AQI:** `155` (Unhealthy)

**Case B: High Heat, Lower Humidity**
*Input:* `PM10: 67`, `NO2: 60`, `CO: 680`, `Temp: 39°C`, `Humidity: 49%`

> **Result:** Despite high PM10 inputs, the model recognizes that higher temperatures can aid dispersion, resulting in a slightly lower risk category.
>
>   * **Predicted PM2.5:** `54.99 μg/m³`
>   * **AQI:** `149` (Unhealthy for Sensitive Groups)

-----

## 🚀 Key Features

  * **🌍 Real-Time Monitoring:** Live tracking of AQI, PM2.5, PM10, and meteorological conditions using the OpenWeatherMap API.
  * **🔮 48-Hour Forecasting:** Machine Learning pipeline to predict smog risk hours up to 48 hours in advance.
  * **🤖 AI Health Assistant:** A **RAG-powered chatbot** (Weaviate & Groq LLaMA 3) that answers user queries based on a scientific knowledge base.
  * **📊 Interactive Dashboard:** A responsive React frontend featuring data visualization (Recharts) and dynamic smog alert indicators.
  * **📈 Custom Predictive Tool:** Users can manually input pollutant/weather variables to simulate AQI scenarios (as seen in the metrics above).
  * **🐳 Fully Dockerized:** Seamless deployment and orchestration using Docker and Docker Compose.

-----

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend** | React.js (Hooks, Context), Tailwind CSS, Recharts |
| **Backend** | FastAPI, Uvicorn, Python 3.9 |
| **ML Engine** | **Scikit-Learn (Random Forest)**, Joblib (Parallelization), Pandas |
| **AI & RAG** | Weaviate (Vector DB), Sentence Transformers, Groq API (LLaMA 3) |
| **DevOps** | Docker, Docker Compose, Nginx |

-----

## ⚡ Quick Start

### 1\. Prerequisites

  * Docker & Docker Compose
  * Git

### 2\. Clone & Configure

```bash
git clone https://github.com/Ashir-Ali-Shah/aqi-quality-project.git
cd aqi-quality-project
```

*Update `docker-compose.yml` with your `GROQ_API_KEY` and `OPENWEATHER_API_KEY`.*

### 3\. Build & Run

```bash
docker-compose up --build
```

### 4\. Access Points

  * 🖥️ **Dashboard:** `http://localhost:3000`
  * 📄 **API Docs:** `http://localhost:8000/docs`
  * 🧠 **Vector DB:** `http://localhost:8080`

-----

## 📂 Project Structure

```bash
air-quality-project/
├── docker-compose.yml       # Orchestration
├── backend/                 # FastAPI & ML Logic
│   ├── backend.py           # API Endpoints
│   ├── ml_pipeline.py       # Random Forest Implementation
│   ├── scaler.pkl           # Pre-trained Feature Scaler
│   └── Dockerfile           # Backend Config
└── frontend/                # React Dashboard
    ├── src/
    │   ├── SmogSentinelTabs.js # Visualization Logic
    └── Dockerfile           # Frontend Config
```
