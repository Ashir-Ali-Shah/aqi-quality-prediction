# Urban Air Quality Sentinel

**A Full-Stack AI System for Real-time Smog Detection, Forecasting, and Health Assistance.**

Air Quality Sentinel is a complete application that tracks air quality in major cities. It features an advanced AI Health Assistant, a fast machine learning forecasting engine, and a live data dashboard.

## The Advanced RAG AI Assistant

Our system features a highly robust Retrieval-Augmented Generation (RAG) architecture. This acts as the brain of the application, giving users instant, scientifically accurate health advice based on real-time air quality. 

### How It Works

The AI system is built on three core components:
1. **The Static Knowledge:** Health and science articles are stored securely in a Weaviate vector database.
2. **The Dynamic Context:** Real-time city weather and air quality metrics are fetched continuously.
3. **The Brain:** The Groq LLM (llama-3.1-8b-instant) combines the static health articles with the live weather data to create a concise, up-to-the-minute answer for the user.

### Performance and Evaluation

The entire application runs seamlessly using Docker. We evaluate the AI's performance using Langsmith. Recent evaluation tests scored an impressive **0.85 out of 1.0**. 

Our AI generates text at lightning speed:
* **Generation Speed:** 211.2 Tokens Per Second (TPS).
* **Responsiveness:** 313ms Time-to-First-Token (TTFT).

This means the user gets an ultra-responsive experience with almost no waiting time. There is only a very minor, acceptable delay during the initial database search.

### Graceful Degradation and Stability

To make sure the application never crashes if an external API goes down, we built an impressive 4-tier safety net:
* **Tier 1 (Normal Operation):** Full pipeline using Weaviate and the Groq LLM.
* **Tier 2 (LLM Down):** If the AI generator fails, the system safely displays raw retrieved text snippets from the database.
* **Tier 3 (Database Down):** If the database fails, the system falls back to local Python keyword matching on a backup file.
* **Tier 4 (All Systems Down):** Returns a clean and polite "Service unavailable" message to the user.

We also use Redis caching and background pre-fetching. This keeps the application lightning fast and reduces API costs.

### Technical Architecture Insights

When designing this system, several key engineering choices were made:
* **Hybrid Search and Cross-Encoders:** We combine the exactness of keyword search with the context of semantic search. The cross-encoder then guarantees that only the most relevant document gets passed to the AI.
* **Handling Latency and Rate Limits:** We handle this through our safety net tiers, Redis caching for common questions, and continuous background data prefetching.

## Project Features and Forecasting

While the RAG AI is the star of the show, the rest of the project is equally robust.

### Machine Learning Forecasting
Instead of using slow and expensive LLMs for predicting numerical data, we use standard Machine Learning models like Random Forest. Traditional ML is much better, faster, and cheaper for time-series forecasting. Our Random Forest Regressor provides 48-hour smog risk predictions with ultra-low latency (around 0.1 seconds per request).

### Live Dashboard
* **Real-Time Monitoring:** Live tracking of AQI, PM2.5, PM10, and weather using the OpenWeatherMap API.
* **Interactive Interface:** A responsive React frontend featuring data visualizations with Recharts and dynamic smog alerts.
* **Custom Tools:** Users can manually input pollutant variables to simulate different air quality scenarios.

## Tech Stack

* **Frontend:** React.js, Tailwind CSS, Recharts
* **Backend:** FastAPI, Python 3.9
* **ML Engine:** Scikit-Learn (Random Forest)
* **AI and Database:** Weaviate, Groq API (LLaMA 3)
* **DevOps:** Docker, Docker Compose, Redis

## Quick Start

1. Clone the repository.
2. Update the `docker-compose.yml` file with your API keys.
3. Run `docker compose up --build -d`.
4. Access the dashboard at `http://localhost:3000`.
