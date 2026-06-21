"""
Enhanced FastAPI Backend with RAG Implementation using Weaviate
Real-time air quality data from OpenWeatherMap + Weaviate vector store + Sentence Transformers + Groq LLM
Features: Redis FAQ caching, Graceful Degradation, Concise RAG output
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
import os
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
from collections import deque
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')
import joblib
import redis
import requests
import socket
from langsmith import traceable

# Monkey-patch socket to force IPv4 to prevent 5-second Docker Desktop IPv6 DNS timeouts
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv4

import os
os.environ["GRPC_DNS_RESOLVER"] = "native"

# ML and RAG imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer, CrossEncoder
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# ============================================================================
# PM2.5 Prediction Model Configuration
# ============================================================================

MODEL_PATH = 'pm25_random_forest_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
pm25_prediction_model = None
feature_scaler = None

def load_pm25_model():
    """Load the PM2.5 Random Forest model and feature scaler"""
    global pm25_prediction_model, feature_scaler
    try:
        pm25_prediction_model = joblib.load(MODEL_PATH)
        print(f"✅ PM2.5 Random Forest model loaded successfully from {MODEL_PATH}")

        feature_scaler = joblib.load(SCALER_PATH)
        print(f"✅ Feature scaler loaded successfully from {SCALER_PATH}")

        return True
    except FileNotFoundError as e:
        print(f"⚠️ Model or scaler file not found:")
        print(f" - Model: {MODEL_PATH}")
        print(f" - Scaler: {SCALER_PATH}")
        print(f" Error: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️ Error loading model or scaler: {str(e)}")
        return False

def preprocess_prediction_input(input_data):
    """Preprocess input data for the PM2.5 prediction model."""
    required_keys = ['pm10', 'no2', 'o3', 'co', 'so2', 'temperature', 'relative_humidity']

    for key in required_keys:
        if key not in input_data:
            raise ValueError(f"Missing required input field: {key}")

    features = [
        float(input_data['pm10']),
        float(input_data['no2']),
        float(input_data['o3']),
        float(input_data['co']),
        float(input_data['so2']),
        float(input_data['temperature']),
        float(input_data['relative_humidity'])
    ]

    return np.array([features])

def predict_pm25_value(input_data):
    """Use the loaded model and scaler to predict PM2.5."""
    global pm25_prediction_model, feature_scaler

    if pm25_prediction_model is None:
        raise ValueError("PM2.5 prediction model not loaded. Please check model file.")
    if feature_scaler is None:
        raise ValueError("Feature scaler not loaded. Please check scaler file.")

    processed_data = preprocess_prediction_input(input_data)
    scaled_data = feature_scaler.transform(processed_data)
    prediction = pm25_prediction_model.predict(scaled_data)[0]
    prediction = max(0, prediction)

    return float(prediction)

def calculate_aqi_from_pm25(pm25):
    """Calculate AQI based on PM2.5 value using US EPA formula"""
    if pm25 <= 12:
        return (50 / 12) * pm25
    elif pm25 <= 35.4:
        return 50 + ((100 - 50) / (35.4 - 12)) * (pm25 - 12)
    elif pm25 <= 55.4:
        return 100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4)
    elif pm25 <= 150.4:
        return 150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4)
    elif pm25 <= 250.4:
        return 200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4)
    else:
        return min(300 + ((500 - 300) / (500 - 250.4)) * (pm25 - 250.4), 500)

def get_aqi_category_and_message(aqi):
    """Get AQI category and health message based on AQI value"""
    if aqi <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone. Sensitive groups should avoid outdoor activities."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected. Stay indoors and avoid all outdoor activities."

def calculate_smog_risk(pm25, wind_speed, humidity, pressure, temperature):
    smogScore = 0.0
    factors = []
    
    # 1. Base PM2.5 continuous scoring (e.g., 50 -> 10 pts, 200 -> 40 pts)
    pm25_score = pm25 * 0.20
    smogScore += pm25_score
    if pm25 > 150:
        factors.append(f'Hazardous PM2.5 levels ({pm25:.1f} µg/m³)')
    elif pm25 > 50:
        factors.append(f'Elevated PM2.5 ({pm25:.1f} µg/m³)')
        
    # 2. Wind speed penalty: Inversely proportional (Stagnant air traps smog)
    # E.g., 0 m/s -> 28 pts, 3.5+ m/s -> 0 pts
    wind_score = max(0.0, (3.5 - wind_speed) * 8.0)
    smogScore += wind_score
    if wind_speed < 2.0:
        factors.append(f'Stagnant air (wind: {wind_speed:.1f} m/s)')
            
    # 3. Humidity penalty: Linear increase above 50%
    # E.g., 50% -> 0 pts, 90% -> 20 pts
    humidity_score = max(0.0, (humidity - 50) * 0.5)
    smogScore += humidity_score
    if humidity > 75:
        factors.append(f'High humidity trapping pollutants ({humidity:.1f}%)')
            
    # 4. Temperature inversion risk (Compound factor: High humidity + Low Wind)
    if wind_speed < 2.0 and humidity > 70:
        inversion_multiplier = ((2.0 - wind_speed) / 2.0) * ((humidity - 70) / 30.0) 
        inversion_penalty = inversion_multiplier * 30.0  # Up to 30 extra points
        smogScore += inversion_penalty
        factors.append('Conditions favorable for temperature inversion')

    # 5. Pressure: High pressure systems trap air
    pressure_score = max(0.0, (pressure - 1010) * 1.0)
    smogScore += min(15.0, pressure_score)  # Cap pressure impact at 15
        
    # 6. Temperature: Cold air can trap pollutants
    temp_score = max(0.0, (20 - temperature) * 0.5)
    smogScore += temp_score
        
    # Cap score at 100
    smogScore = min(smogScore, 100.0)
        
    if smogScore >= 80:
        severity = 'EXTREME'
        color = '#7f1d1d'
        actions = ['Stay indoors at all times', 'Use air purifiers continuously', 'Wear N95 masks if must go outside']
    elif smogScore >= 60:
        severity = 'SEVERE'
        color = '#991b1b'
        actions = ['Minimize outdoor activities', 'Use air purifiers', 'Wear N95 masks outdoors']
    elif smogScore >= 40:
        severity = 'HIGH'
        color = '#dc2626'
        actions = ['Limit outdoor exposure', 'Sensitive groups stay indoors', 'Consider wearing masks']
    elif smogScore >= 20:
        severity = 'MODERATE'
        color = '#f97316'
        actions = ['Sensitive individuals limit prolonged outdoor activities', 'Monitor air quality']
    else:
        severity = 'LOW'
        color = '#10b981'
        actions = ['Air quality is acceptable', 'Normal activities can continue']
        
    return {
        'smog_score': float(smogScore),
        'smog_severity': severity,
        'smog_probability': float(smogScore),
        'smog_color': color,
        'smog_factors': factors,
        'smog_actions': actions
    }

# ============================================================================
# Pydantic Models - Define BEFORE FastAPI app
# ============================================================================

class PredictionInput(BaseModel):
    pm10: float
    no2: float
    o3: float
    co: float
    so2: float
    temperature: float
    relative_humidity: float

class PredictionResponse(BaseModel):
    pm25_prediction: float
    aqi: float
    health_category: str
    health_message: str
    input_data: Dict

class AirQualityResponse(BaseModel):
    city: str
    timestamp: str
    pm25: float
    pm10: float
    no2: float
    o3: float
    so2: float
    co: float
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    pressure: float
    visibility: float
    aqi: float
    is_smog_emergency: bool
    smog_score: float
    smog_severity: str
    smog_probability: float
    smog_color: str
    smog_factors: List[str]
    smog_actions: List[str]
    smog_trend: str

class RAGQueryRequest(BaseModel):
    question: str
    city: Optional[str] = "Islamabad"
    language: str = "en"
    top_k: int = 3

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    source_ids: List[str]
    similarity_scores: List[float]
    timestamp: str
    current_data: Optional[Dict] = None

class ForecastPrediction(BaseModel):
    hour: int
    timestamp: str
    predicted_pm25: float
    predicted_aqi: float
    smog_likely: bool
    smog_probability: float
    confidence: float
    temperature: float
    humidity: float
    wind_speed: float
    pressure: float

class ForecastResponse(BaseModel):
    predictions: List[ForecastPrediction]
    summary: str
    smog_hours: int
    peak_pm25: float
    peak_aqi: float
    peak_hour: int
    average_confidence: float

# ============================================================================
# Initialize FastAPI
# ============================================================================

app = FastAPI(
    title="Urban Air Quality Sentinel API with Weaviate RAG",
    description="Real-time smog detection, forecasting, and Weaviate-powered RAG insights",
    version="3.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"

# Weaviate Configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY', None)

# Pakistani cities with coordinates
PAKISTAN_CITIES = {
    "Lahore": {"lat": 31.5497, "lon": 74.3436},
    "Karachi": {"lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Rawalpindi": {"lat": 33.5651, "lon": 73.0169},
    "Faisalabad": {"lat": 31.4180, "lon": 73.0790},
    "Multan": {"lat": 30.1575, "lon": 71.5249},
    "Peshawar": {"lat": 34.0151, "lon": 71.5249},
    "Quetta": {"lat": 30.1798, "lon": 66.9750},
    "Sialkot": {"lat": 32.4945, "lon": 74.5229},
    "Gujranwala": {"lat": 32.1617, "lon": 74.1883}
}

# Historical data storage
historical_data = deque(maxlen=1000)
cache = {}
CACHE_TTL = 300

# ============================================================================
# RAG Knowledge Base
# ============================================================================

KNOWLEDGE_BASE = [
    {
        'id': 'pm25_basics',
        'category': 'pollutants',
        'title': 'What is PM2.5?',
        'content': 'PM2.5 refers to fine particulate matter with a diameter of 2.5 micrometers or less. These particles are small enough to penetrate deep into the lungs and even enter the bloodstream, causing serious health issues including respiratory diseases, cardiovascular problems, and premature death. Common sources include vehicle emissions, industrial activities, construction dust, and biomass burning.',
        'keywords': ['pm2.5', 'particulate matter', 'fine particles', 'pollution', 'definition']
    },
    {
        'id': 'smog_formation',
        'category': 'science',
        'title': 'How Smog Forms',
        'content': 'Smog forms when pollutants like PM2.5, nitrogen oxides, and volatile organic compounds accumulate in the atmosphere under specific weather conditions. Key factors include temperature inversion, low wind speeds, high humidity, and increased emissions from traffic and industry.',
        'keywords': ['smog', 'formation', 'causes', 'temperature inversion', 'atmospheric conditions']
    },
    {
        'id': 'health_impacts',
        'category': 'health',
        'title': 'Health Effects of Air Pollution',
        'content': 'Short-term exposure to high PM2.5 levels causes eye irritation, coughing, breathing difficulties, and asthma attacks. Long-term exposure increases risks of chronic bronchitis, reduced lung function, heart disease, stroke, lung cancer, and premature death.',
        'keywords': ['health', 'effects', 'impacts', 'symptoms', 'risks', 'diseases']
    },
    {
        'id': 'protection_measures',
        'category': 'safety',
        'title': 'How to Protect Yourself',
        'content': 'During high pollution: Stay indoors with windows closed. Use N95 masks if going outside. Run air purifiers with HEPA filters. Avoid outdoor exercise during peak traffic hours. Monitor air quality apps regularly.',
        'keywords': ['protection', 'safety', 'masks', 'air purifiers', 'precautions']
    }
]

# ============================================================================
# Weaviate RAG System
# ============================================================================

class WeaviateRAGSystem:
    """Advanced RAG system using Weaviate + Sentence Transformers + Groq"""

    def __init__(self, knowledge_base: List[Dict], model_name: str = 'all-MiniLM-L6-v2'):
        self.knowledge_base = knowledge_base
        self.model_name = model_name
        self.embedding_model = None
        self.cross_encoder = None
        self.weaviate_client = None
        self.collection_name = "AirQualityKnowledge"
        self.is_initialized = False

    def initialize(self):
        """Initialize embedding model and Weaviate connection"""
        print("🔧 Initializing Weaviate RAG System...")

        print(f" Loading embedding model: {self.model_name}...")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        print(f" Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print(f" Connecting to Weaviate at {WEAVIATE_URL}...")
        try:
            if WEAVIATE_API_KEY:
                self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=WEAVIATE_URL,
                    auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
                )
            else:
                host_str = WEAVIATE_URL.replace('http://', '').replace('https://', '').split(':')[0]
                try:
                    ip_addr = socket.gethostbyname(host_str)
                except:
                    ip_addr = host_str
                self.weaviate_client = weaviate.connect_to_local(
                    host=ip_addr,
                    port=int(WEAVIATE_URL.split(':')[-1]) if ':' in WEAVIATE_URL.split('//')[-1] else 8080
                )

            print(" ✅ Connected to Weaviate successfully!")

        except Exception as e:
            print(f" ❌ Failed to connect to Weaviate: {e}")
            raise Exception(f"Weaviate connection failed: {e}")

        self._create_collection_schema()
        self._index_documents()

        self.is_initialized = True
        print(f"✅ Weaviate RAG System initialized with {len(self.knowledge_base)} documents!")

    def _create_collection_schema(self):
        """Create or recreate Weaviate collection schema"""
        print(" Creating Weaviate collection schema...")

        try:
            if self.weaviate_client.collections.exists(self.collection_name):
                print(f" Deleting existing collection: {self.collection_name}")
                self.weaviate_client.collections.delete(self.collection_name)

            from weaviate.classes.config import Property, DataType, Configure

            self.weaviate_client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                ]
            )

            print(f" ✅ Collection '{self.collection_name}' created successfully!")

        except Exception as e:
            print(f" ❌ Error creating collection: {e}")
            raise

    def _index_documents(self):
        """Index all documents in Weaviate with embeddings"""
        print(f" Indexing {len(self.knowledge_base)} documents...")

        collection = self.weaviate_client.collections.get(self.collection_name)

        documents_data = []
        texts_for_embedding = []

        for doc in self.knowledge_base:
            text = f"{doc['title']}. {doc['content']} Keywords: {', '.join(doc['keywords'])}"
            texts_for_embedding.append(text)

            documents_data.append({
                'doc_id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                'category': doc['category'],
                'keywords': doc['keywords']
            })

        print(" Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts_for_embedding,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(" Inserting documents into Weaviate...")
        with collection.batch.dynamic() as batch:
            for doc_data, embedding in zip(documents_data, embeddings):
                batch.add_object(
                    properties=doc_data,
                    vector=embedding.tolist()
                )

        print(f" ✅ Successfully indexed {len(documents_data)} documents!")

    @traceable(name="expand_query", run_type="llm")
    def _expand_query(self, query: str) -> str:
        """Expand query using Groq API to include synonyms and related terms for better hybrid search."""
        system_prompt = "You are an expert search query expander. Given a user query about air quality in Pakistan, generate 2-3 related synonyms or variations to improve search recall. Return ONLY the expanded keywords separated by spaces. Do not include the original query, introductions, or explanations."
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Query: {query}"}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 30
                },
                timeout=10.0
            )
            if response.status_code == 200:
                result = response.json()
                expanded = result['choices'][0]['message']['content'].strip()
                print(f"🔄 Query Expanded: '{query}' -> '{expanded}'")
                return f"{query} {expanded}"
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            
        return query

    @traceable(name="retrieve_relevant", run_type="retriever")
    def retrieve_relevant(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant documents using Hybrid Search + Cross-Encoder Reranking"""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Call initialize() first.")

        # Step 1: Query Expansion
        expanded_query = self._expand_query(query)

        # Step 2: Hybrid Search (Retrieve broader set of candidates)
        expanded_k = max(10, top_k * 3)
        query_embedding = self.embedding_model.encode(
            [query],  # Embed the original query, not the expanded one
            convert_to_numpy=True
        )[0]

        collection = self.weaviate_client.collections.get(self.collection_name)

        response = collection.query.hybrid(
            query=expanded_query,
            vector=query_embedding.tolist(),
            limit=expanded_k,
            alpha=0.5,
            return_metadata=MetadataQuery(score=True)
        )

        candidates = []
        for obj in response.objects:
            candidates.append({
                'id': obj.properties['doc_id'],
                'title': obj.properties['title'],
                'content': obj.properties['content'],
                'category': obj.properties['category'],
                'keywords': obj.properties['keywords'],
                'hybrid_score': obj.metadata.score if obj.metadata.score else 0.0
            })

        if not candidates:
            return []

        # Step 3: Cross-Encoder Reranking
        pairs = [[query, doc['content']] for doc in candidates]
        cross_scores = self.cross_encoder.predict(pairs)

        for doc, score in zip(candidates, cross_scores):
            doc['similarity_score'] = float(score)

        # Step 4: Sort by cross-encoder score and keep top_k
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        return candidates[:top_k]

    @traceable(name="generate_answer", run_type="llm")
    async def generate_answer(
        self,
        query: str,
        context_docs: List[Dict],
        current_data: Optional[Dict] = None,
        language: str = 'en',
        all_cities_data: Optional[List[Dict]] = None
    ) -> Dict:
        """Generate answer using Groq LLM with retrieved context — concise output to save tokens"""

        system_prompt = """You are an expert environmental scientist specializing in air quality and public health in Pakistan.
Use the provided context to answer questions accurately.
CRITICAL RULES:
- Answer in 1-2 sentences MAXIMUM. Be extremely concise.
- No introductions, no filler, no repetition.
- Note that Islamabad, Lahore, Pindi (Rawalpindi) etc. are CITIES in Pakistan.
- Include specific numbers from the data when relevant.
- Provide one actionable recommendation if applicable.
- Do not hallucinate."""

        user_prompt = "Context from knowledge base:\n\n"
        for idx, doc in enumerate(context_docs, 1):
            user_prompt += f"[Source {idx}: {doc['title']}]\n{doc['content']}\n\n"

        if all_cities_data:
            user_prompt += "\nCurrent Real-time Data for mentioned cities:\n"
            for data in all_cities_data:
                user_prompt += f"- {data.get('city')}: PM2.5: {data.get('pm25', 0):.1f} µg/m³ | AQI: {data.get('aqi', 0):.0f} | Temp: {data.get('temperature', 0):.1f}°C\n"
            user_prompt += "\n"
        elif current_data:
            user_prompt += f"\nCurrent Real-time Data for {current_data.get('city', 'Unknown')}:\n"
            user_prompt += f"- PM2.5: {current_data.get('pm25', 0):.1f} µg/m³\n"
            user_prompt += f"- AQI: {current_data.get('aqi', 0):.0f}\n"
            user_prompt += f"- Temperature: {current_data.get('temperature', 0):.1f}°C\n"
            user_prompt += f"- Humidity: {current_data.get('humidity', 0):.0f}%\n\n"

        user_prompt += f"Question: {query}\n\nProvide a very concise answer (1-2 sentences only) based on the context and current data."

        try:
            def sync_call():
                return requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 100,
                        "top_p": 0.9
                    },
                    timeout=20.0
                )
                
            response = await asyncio.to_thread(sync_call)

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']

                return {
                    'answer': answer,
                    'sources': [doc['title'] for doc in context_docs],
                    'source_ids': [doc['id'] for doc in context_docs],
                    'similarity_scores': [doc.get('similarity_score', 0) for doc in context_docs],
                    'success': True
                }
            else:
                raise Exception(f"Groq API error: {response.status_code}")

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': 'I apologize, but I encountered an error generating a response. Please try again.',
                'sources': [],
                'source_ids': [],
                'similarity_scores': [],
                'success': False,
                'error': str(e)
            }

    def close(self):
        """Close Weaviate connection"""
        if self.weaviate_client:
            self.weaviate_client.close()
            print("✅ Weaviate connection closed")

# Initialize RAG system globally
rag_system = WeaviateRAGSystem(KNOWLEDGE_BASE)

# ============================================================================
# Redis Cache Service (Single Responsibility: FAQ Query Caching)
# ============================================================================

class RedisCacheService:
    """Handles caching of frequently asked RAG queries using Redis.
    Follows Single Responsibility Principle — only manages cache operations.
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.client = None
        self.is_connected = False
        self.default_ttl = 3600  # 1 hour TTL for cached FAQ responses

    def connect(self):
        """Establish connection to Redis"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.client.ping()
            self.is_connected = True
            print(f"✅ Redis connected at {self.redis_url}")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}. Continuing without cache.")
            self.is_connected = False

    def _normalize_query(self, question: str, city: str = "") -> str:
        """Normalize query for consistent cache keys.
        Lowercases, strips whitespace, sorts words for order-independent matching.
        """
        normalized = question.lower().strip()
        # Sort words for order-independent matching
        words = sorted(normalized.split())
        key_string = f"{' '.join(words)}:{city.lower().strip()}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_cached_response(self, question: str, city: str = "") -> Optional[Dict]:
        """Retrieve cached RAG response if available"""
        if not self.is_connected:
            return None
        try:
            cache_key = f"rag_cache:{self._normalize_query(question, city)}"
            cached = self.client.get(cache_key)
            if cached:
                print(f"✅ Redis CACHE HIT for query: '{question[:50]}...'")
                return json.loads(cached)
            return None
        except Exception as e:
            print(f"⚠️ Redis GET error: {e}")
            return None

    def cache_response(self, question: str, city: str, response: Dict, ttl: int = None) -> bool:
        """Cache a RAG response in Redis"""
        if not self.is_connected:
            return False
        try:
            cache_key = f"rag_cache:{self._normalize_query(question, city)}"
            self.client.setex(
                cache_key,
                ttl or self.default_ttl,
                json.dumps(response)
            )
            print(f"✅ Redis CACHED response for query: '{question[:50]}...'")
            return True
        except Exception as e:
            print(f"⚠️ Redis SET error: {e}")
            return False

    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            print("✅ Redis connection closed")

# Initialize Redis cache service globally
redis_cache = RedisCacheService()

# ============================================================================
# Graceful Degradation Service (Single Responsibility: Fallback Logic)
# ============================================================================

class GracefulDegradationService:
    """Provides tiered fallback when APIs fail.
    Follows Single Responsibility Principle — only manages degradation logic.
    
    Tiers:
      1. Full RAG pipeline (Weaviate + Groq LLM)
      2. Weaviate-only (return retrieved docs without LLM)
      3. Local keyword search against KNOWLEDGE_BASE
      4. "Service not available, try again later!"
    """

    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base

    def keyword_search(self, query: str, top_k: int = 2) -> List[Dict]:
        """Tier 3: Local keyword-based search when Weaviate is unavailable"""
        query_tokens = query.lower().split()
        scored = []

        for doc in self.knowledge_base:
            score = 0
            doc_text = (doc['content'] + ' ' + doc['title'] + ' ' + ' '.join(doc['keywords'])).lower()
            for token in query_tokens:
                if any(token in kw for kw in doc['keywords']):
                    score += 3
                if token in doc['title'].lower():
                    score += 2
                score += doc_text.count(token) * 0.5
            if score > 0:
                scored.append({**doc, 'score': score})

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    def synthesize_from_docs(self, docs: List[Dict], query: str) -> str:
        """Tier 2: Create a concise answer from retrieved documents without LLM"""
        if not docs:
            return self.service_unavailable_message()

        # Pick the most relevant document and return a trimmed snippet
        best_doc = docs[0]
        content = best_doc['content']
        # Return first 2 sentences for conciseness
        sentences = content.split('.')
        concise = '. '.join(sentences[:2]).strip()
        if concise and not concise.endswith('.'):
            concise += '.'
        return concise

    def service_unavailable_message(self) -> str:
        """Tier 4: Final fallback message"""
        return "Service not available, try again later!"

    @traceable(name="execute_with_fallback", run_type="chain")
    async def execute_with_fallback(
        self,
        rag_system,
        query: str,
        city: str,
        top_k: int,
        current_data: Optional[Dict] = None,
        all_cities_data: Optional[List[Dict]] = None,
        language: str = 'en'
    ) -> Dict:
        """Execute RAG query with tiered graceful degradation"""

        # --- Tier 1: Full RAG pipeline (Weaviate + Groq) ---
        try:
            import time
            import logging
            if rag_system.is_initialized:
                t0 = time.time()
                relevant_docs = rag_system.retrieve_relevant(query, top_k=top_k)
                t1 = time.time(); open('/tmp/rag_timing.log', 'a').write(f"TIMING RAG TIER1: retrieve_relevant took {t1-t0:.3f}s\n")
                if relevant_docs:
                    result = await rag_system.generate_answer(
                        query, relevant_docs, current_data, language, all_cities_data
                    )
                    t2 = time.time(); open('/tmp/rag_timing.log', 'a').write(f"TIMING RAG TIER1: generate_answer took {t2-t1:.3f}s\n")
                    if result.get('success'):
                        return {
                            'answer': result['answer'],
                            'sources': result['sources'],
                            'source_ids': result['source_ids'],
                            'similarity_scores': result['similarity_scores'],
                            'tier': 1
                        }
                    # Groq failed but we have docs — fall to Tier 2
                    print("⚠️ Tier 1 failed (LLM error). Falling back to Tier 2.")
                    answer = self.synthesize_from_docs(relevant_docs, query)
                    return {
                        'answer': answer,
                        'sources': [doc['title'] for doc in relevant_docs],
                        'source_ids': [doc['id'] for doc in relevant_docs],
                        'similarity_scores': [doc.get('similarity_score', 0) for doc in relevant_docs],
                        'tier': 2
                    }
        except Exception as e:
            print(f"⚠️ Tier 1 & 2 failed: {e}. Falling back to Tier 3.")

        # --- Tier 3: Local keyword search ---
        try:
            local_docs = self.keyword_search(query, top_k=2)
            if local_docs:
                answer = self.synthesize_from_docs(local_docs, query)
                return {
                    'answer': answer,
                    'sources': [doc['title'] for doc in local_docs],
                    'source_ids': [doc['id'] for doc in local_docs],
                    'similarity_scores': [],
                    'tier': 3
                }
        except Exception as e:
            print(f"⚠️ Tier 3 failed: {e}. Falling back to Tier 4.")

        # --- Tier 4: Service unavailable ---
        return {
            'answer': self.service_unavailable_message(),
            'sources': [],
            'source_ids': [],
            'similarity_scores': [],
            'tier': 4
        }

# Initialize graceful degradation service globally
degradation_service = GracefulDegradationService(KNOWLEDGE_BASE)

# ============================================================================
# ML Pipeline
# ============================================================================

class SmogMLPipeline:
    """ML pipeline for PM2.5 forecasting and smog classification"""

    def __init__(self):
        self.pm25_model = None
        self.smog_classifier = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'pm25_current', 'pm10', 'no2', 'o3', 'so2', 'co',
            'temperature', 'humidity', 'wind_speed', 'pressure',
            'hour', 'month', 'wind_direction', 'pm25_trend'
        ]
        self.is_trained = False

    def create_training_data(self, n_samples=5000):
        """Generate realistic training data"""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            hour = np.random.randint(0, 24)
            month = np.random.randint(1, 13)
            is_winter = month in [11, 12, 1, 2]
            is_rush_hour = hour in [7, 8, 9, 17, 18, 19, 20]

            base_pm25 = np.random.uniform(100, 300) if is_winter else np.random.uniform(30, 100)
            if is_rush_hour:
                base_pm25 *= np.random.uniform(1.5, 2.0)

            pm25 = max(0, base_pm25 + np.random.normal(0, 20))

            data.append({
                'pm25_current': pm25,
                'pm10': pm25 * np.random.uniform(1.8, 2.5),
                'no2': pm25 * np.random.uniform(0.4, 0.8),
                'o3': np.random.uniform(30, 80),
                'so2': pm25 * np.random.uniform(0.2, 0.5),
                'co': pm25 * np.random.uniform(300, 800),
                'temperature': np.random.uniform(5, 18) if pm25 > 200 else np.random.uniform(15, 35),
                'humidity': np.random.uniform(70, 95) if pm25 > 200 else np.random.uniform(30, 70),
                'wind_speed': np.random.uniform(0.5, 2.0) if pm25 > 200 else np.random.uniform(2.0, 6.0),
                'pressure': np.random.uniform(1010, 1025),
                'wind_direction': np.random.uniform(0, 360),
                'pm25_trend': np.random.uniform(-50, 50),
                'hour': hour,
                'month': month,
                'aqi': calculate_aqi_from_pm25(pm25),
                'is_smog': int(pm25 > 250 and np.random.uniform(0.5, 2.0) < 2.0)
            })

        return pd.DataFrame(data)

    def train_models(self):
        """Train forecasting and classification models"""
        print("🔧 Training ML models...")
        df = self.create_training_data(5000)

        X = df[self.feature_columns].fillna(0)
        y_aqi = df['aqi']
        y_smog = df['is_smog']

        X_scaled = self.scaler.fit_transform(X)

        self.pm25_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.pm25_model.fit(X_scaled, y_aqi)

        self.smog_classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.smog_classifier.fit(X_scaled, y_smog)

        self.is_trained = True
        print("✅ ML models trained successfully!")

    def predict_48h(self, current_data, weather_forecast):
        """Generate 48-hour predictions"""
        if not self.is_trained:
            raise ValueError("Models not trained")

        predictions = []
        current_time = datetime.now()

        for hour_offset in range(48):
            forecast_time = current_time + timedelta(hours=hour_offset)

            weather = weather_forecast[hour_offset] if hour_offset < len(weather_forecast) else weather_forecast[-1]

            pm25_decay = max(0.7, 1.0 - (hour_offset * 0.01))
            features = {
                'pm25_current': current_data['pm25'] * pm25_decay,
                'pm10': current_data['pm10'] * pm25_decay,
                'no2': current_data['no2'] * pm25_decay,
                'o3': current_data.get('o3', 50),
                'so2': current_data.get('so2', 10),
                'co': current_data.get('co', 500),
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'wind_speed': weather['wind_speed'],
                'pressure': weather['pressure'],
                'wind_direction': weather.get('wind_direction', 180),
                'pm25_trend': 0,
                'hour': forecast_time.hour,
                'month': forecast_time.month
            }

            df = pd.DataFrame([features])
            X = df[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)

            predicted_aqi = float(self.pm25_model.predict(X_scaled)[0])
            predicted_pm25 = self._aqi_to_pm25(predicted_aqi)
            smog_proba = self.smog_classifier.predict_proba(X_scaled)[0]

            predictions.append({
                'hour': hour_offset,
                'timestamp': forecast_time.isoformat(),
                'predicted_pm25': predicted_pm25,
                'predicted_aqi': predicted_aqi,
                'smog_likely': bool(smog_proba[1] > 0.5),
                'smog_probability': float(smog_proba[1]),
                'confidence': float(max(smog_proba)),
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'wind_speed': weather['wind_speed'],
                'pressure': weather['pressure']
            })

        return predictions

    def _aqi_to_pm25(self, aqi):
        """Convert AQI back to PM2.5"""
        if aqi <= 50:
            return (aqi / 50) * 12
        elif aqi <= 100:
            return 12 + ((aqi - 50) / 50) * (35.4 - 12)
        elif aqi <= 150:
            return 35.4 + ((aqi - 100) / 50) * (55.4 - 35.4)
        elif aqi <= 200:
            return 55.4 + ((aqi - 150) / 50) * (150.4 - 55.4)
        elif aqi <= 300:
            return 150.4 + ((aqi - 200) / 100) * (250.4 - 150.4)
        else:
            return 250.4 + ((aqi - 300) / 200) * (500 - 250.4)

ml_pipeline = SmogMLPipeline()

# ============================================================================
# Helper Functions
# ============================================================================

async def fetch_air_quality_data(lat: float, lon: float) -> Dict:
    """Fetch real-time air quality from OpenWeatherMap"""
    cache_key = f"air_quality:{lat}:{lon}"

    if cache_key in cache:
        entry = cache[cache_key]
        if (datetime.now().timestamp() - entry['timestamp']) < CACHE_TTL:
            return entry['data']

    url = f"{OPENWEATHER_BASE}/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    try:
        response = await asyncio.to_thread(requests.get, url, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        
        cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now().timestamp()
        }
        return data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch air quality data: {str(e)}")

async def fetch_air_quality_forecast(lat: float, lon: float) -> Dict:
    """Fetch air quality forecast from OpenWeatherMap"""
    cache_key = f"air_forecast:{lat}:{lon}"
    if cache_key in cache:
        entry = cache[cache_key]
        if (datetime.now().timestamp() - entry['timestamp']) < CACHE_TTL:
            return entry['data']

    url = f"{OPENWEATHER_BASE}/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        cache[cache_key] = {'data': data, 'timestamp': datetime.now().timestamp()}
        return data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch air quality forecast: {str(e)}")

async def fetch_weather_data(lat: float, lon: float) -> Dict:
    """Fetch weather data from OpenWeatherMap"""
    cache_key = f"weather:{lat}:{lon}"
    if cache_key in cache:
        entry = cache[cache_key]
        if (datetime.now().timestamp() - entry['timestamp']) < CACHE_TTL:
            return entry['data']

    url = f"{OPENWEATHER_BASE}/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        cache[cache_key] = {'data': data, 'timestamp': datetime.now().timestamp()}
        return data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch weather data: {str(e)}")

async def fetch_weather_forecast(lat: float, lon: float) -> Dict:
    """Fetch weather forecast from OpenWeatherMap"""
    cache_key = f"weather_forecast:{lat}:{lon}"
    if cache_key in cache:
        entry = cache[cache_key]
        if (datetime.now().timestamp() - entry['timestamp']) < CACHE_TTL:
            return entry['data']

    url = f"{OPENWEATHER_BASE}/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        cache[cache_key] = {'data': data, 'timestamp': datetime.now().timestamp()}
        return data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch weather forecast: {str(e)}")

def parse_responses(air_data: Dict, weather_data: Dict, air_forecast: Dict, weather_forecast: Dict) -> Dict:
    """Parse OpenWeatherMap responses"""
    try:
        current_air = air_data['list'][0]['components'] if air_data.get('list') else {}
        current_weather = weather_data

        hourly_pm25 = [item['components']['pm2_5'] for item in air_forecast.get('list', [])]
        hourly_temperature = []
        hourly_humidity = []
        hourly_wind_speed = []
        hourly_pressure = []
        hourly_wind_direction = []

        wf_list = weather_forecast.get('list', [])
        for i in range(48):
            step = min(i // 3, len(wf_list) - 1)
            if step < len(wf_list):
                item = wf_list[step]
                hourly_temperature.append(item['main']['temp'])
                hourly_humidity.append(item['main']['humidity'])
                hourly_wind_speed.append(item['wind']['speed'])
                hourly_pressure.append(item['main']['pressure'])
                hourly_wind_direction.append(item['wind']['deg'])
            else:
                hourly_temperature.append(hourly_temperature[-1] if hourly_temperature else 20)
                hourly_humidity.append(hourly_humidity[-1] if hourly_humidity else 60)
                hourly_wind_speed.append(hourly_wind_speed[-1] if hourly_wind_speed else 2)
                hourly_pressure.append(hourly_pressure[-1] if hourly_pressure else 1013)
                hourly_wind_direction.append(hourly_wind_direction[-1] if hourly_wind_direction else 180)

        return {
            'pm25': current_air.get('pm2_5', 0),
            'pm10': current_air.get('pm10', 0),
            'no2': current_air.get('no2', 0),
            'o3': current_air.get('o3', 0),
            'so2': current_air.get('so2', 0),
            'co': current_air.get('co', 0),
            'temperature': current_weather['main'].get('temp', 20),
            'humidity': current_weather['main'].get('humidity', 60),
            'wind_speed': current_weather['wind'].get('speed', 2),
            'pressure': current_weather['main'].get('pressure', 1013),
            'visibility': current_weather.get('visibility', 10000),
            'wind_direction': current_weather['wind'].get('deg', 180),
            'hourly_forecast': {
                'pm25': hourly_pm25[:48],
                'temperature': hourly_temperature,
                'humidity': hourly_humidity,
                'wind_speed': hourly_wind_speed,
                'pressure': hourly_pressure,
                'wind_direction': hourly_wind_direction
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing data: {str(e)}")

# ============================================================================
# Background Tasks
# ============================================================================

async def prefetch_all_city_data():
    """Periodically fetch and cache data for all cities in the background"""
    while True:
        print("🔄 Pre-fetching data for all cities to ensure <300ms latency...")
        for city, coords in PAKISTAN_CITIES.items():
            try:
                # Force fetch bypassing cache logic temporarily just to update it
                lat, lon = coords['lat'], coords['lon']
                
                # 1. Current Air Quality
                r1 = await asyncio.to_thread(requests.get, f"{OPENWEATHER_BASE}/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}", timeout=10.0)
                if r1.status_code == 200:
                    cache[f"air_quality:{lat}:{lon}"] = {'data': r1.json(), 'timestamp': datetime.now().timestamp()}
                
                # 2. Weather
                r2 = await asyncio.to_thread(requests.get, f"{OPENWEATHER_BASE}/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric", timeout=10.0)
                if r2.status_code == 200:
                    cache[f"weather:{lat}:{lon}"] = {'data': r2.json(), 'timestamp': datetime.now().timestamp()}
                    
                # 3. Air Forecast
                r3 = await asyncio.to_thread(requests.get, f"{OPENWEATHER_BASE}/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}", timeout=10.0)
                if r3.status_code == 200:
                    cache[f"air_forecast:{lat}:{lon}"] = {'data': r3.json(), 'timestamp': datetime.now().timestamp()}
                    
                # 4. Weather Forecast
                r4 = await asyncio.to_thread(requests.get, f"{OPENWEATHER_BASE}/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric", timeout=10.0)
                if r4.status_code == 200:
                    cache[f"weather_forecast:{lat}:{lon}"] = {'data': r4.json(), 'timestamp': datetime.now().timestamp()}
            except Exception as e:
                print(f"⚠️ Pre-fetch failed for {city}: {e}")
            await asyncio.sleep(1) # stagger requests to avoid rate limits
        
        # Run every 5 minutes
        await asyncio.sleep(300)

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models, Redis cache, and RAG system on startup"""
    print("🚀 Starting Urban Air Quality Sentinel v3.0 with Weaviate RAG + Redis Caching...")

    # Initialize ML models
    ml_pipeline.train_models()

    # Initialize Weaviate RAG system
    try:
        rag_system.initialize()
    except Exception as e:
        print(f"⚠️ WARNING: Failed to initialize Weaviate RAG system: {e}")
        print(" Make sure Weaviate is running. For local instance:")
        print(" docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:latest")

    # Initialize Redis cache for FAQ queries
    redis_cache.connect()

    # Load PM2.5 prediction model
    model_loaded = load_pm25_model()
    if not model_loaded:
        print("⚠️ WARNING: PM2.5 prediction model or scaler not loaded.")
        print(" The /predict_pm25 endpoint will not work.")

    # Start background task for pre-fetching openweather data
    asyncio.create_task(prefetch_all_city_data())

    print("✅ Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown — close Weaviate and Redis connections"""
    print("🛑 Shutting down server...")
    if rag_system.is_initialized:
        rag_system.close()
    redis_cache.close()

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Urban Air Quality Sentinel with Weaviate RAG",
        "version": "3.0.0",
        "features": [
            "Real-time PM2.5 monitoring",
            "48-hour ML forecasting",
            "Weaviate-powered RAG Q&A",
            "PM2.5 prediction from pollutant data",
            "Multi-city coverage"
        ],
        "rag_system": {
            "vector_db": "Weaviate",
            "embedding_model": rag_system.model_name,
            "documents": len(rag_system.knowledge_base),
            "llm": "Groq LLaMA 3.3 70B",
            "initialized": rag_system.is_initialized
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cities")
async def get_cities():
    return {
        "cities": [{"name": name, **coords} for name, coords in PAKISTAN_CITIES.items()],
        "count": len(PAKISTAN_CITIES)
    }

@app.get("/current-air-quality", response_model=AirQualityResponse)
async def get_current_air_quality(
    city: str = Query("Islamabad", description="City name"),
    lat: Optional[float] = None,
    lon: Optional[float] = None
):
    """Get real-time air quality for a city with graceful degradation"""
    try:
        if lat is None or lon is None:
            if city not in PAKISTAN_CITIES:
                raise HTTPException(status_code=404, detail=f"City '{city}' not found")
            coords = PAKISTAN_CITIES[city]
            lat, lon = coords['lat'], coords['lon']

        air_data, weather_data, air_forecast, weather_forecast = await asyncio.gather(
            fetch_air_quality_data(lat, lon),
            fetch_weather_data(lat, lon),
            fetch_air_quality_forecast(lat, lon),
            fetch_weather_forecast(lat, lon)
        )
        parsed = parse_responses(air_data, weather_data, air_forecast, weather_forecast)

        aqi = calculate_aqi_from_pm25(parsed['pm25'])
        is_emergency = parsed['pm25'] > 300 or (parsed['pm25'] > 250 and parsed['wind_speed'] < 2.0)

        risk_data = calculate_smog_risk(
            parsed['pm25'], 
            parsed['wind_speed'], 
            parsed['humidity'], 
            parsed['pressure'], 
            parsed['temperature']
        )
        
        future_pm25 = parsed['hourly_forecast']['pm25'][0] if parsed['hourly_forecast']['pm25'] else parsed['pm25']
        trend = "stable"
        if future_pm25 > parsed['pm25'] * 1.05:
            trend = "rising"
        elif future_pm25 < parsed['pm25'] * 0.95:
            trend = "falling"
        risk_data['smog_trend'] = trend

        historical_data.append({
            'city': city,
            'timestamp': datetime.now().isoformat(),
            **parsed
        })

        # Cache the latest data for graceful degradation fallback
        cache[f"last_good:{city}"] = {
            'data': parsed,
            'aqi': aqi,
            'is_emergency': is_emergency,
            'risk_data': risk_data,
            'timestamp': datetime.now().timestamp()
        }

        return AirQualityResponse(
            city=city,
            timestamp=datetime.now().isoformat(),
            aqi=aqi,
            is_smog_emergency=is_emergency,
            **risk_data,
            **{k: v for k, v in parsed.items() if k != 'hourly_forecast'}
        )

    except HTTPException as he:
        # Graceful degradation: try returning last known good data
        if he.status_code == 503:
            last_good = cache.get(f"last_good:{city}")
            if last_good:
                print(f"⚠️ API unavailable for {city}, returning last cached data")
                risk_fb = last_good.get('risk_data') or calculate_smog_risk(0,0,0,0,0)
                if 'smog_trend' not in risk_fb: risk_fb['smog_trend'] = 'stable'
                return AirQualityResponse(
                    city=city,
                    timestamp=datetime.fromtimestamp(last_good['timestamp']).isoformat(),
                    aqi=last_good['aqi'],
                    is_smog_emergency=last_good['is_emergency'],
                    **risk_fb,
                    **{k: v for k, v in last_good['data'].items() if k != 'hourly_forecast'}
                )
        raise
    except Exception as e:
        # Graceful degradation: try returning last known good data
        last_good = cache.get(f"last_good:{city}")
        if last_good:
            print(f"⚠️ Error for {city}, returning last cached data: {e}")
            risk_fb = last_good.get('risk_data') or calculate_smog_risk(0,0,0,0,0)
            if 'smog_trend' not in risk_fb: risk_fb['smog_trend'] = 'stable'
            return AirQualityResponse(
                city=city,
                timestamp=datetime.fromtimestamp(last_good['timestamp']).isoformat(),
                aqi=last_good['aqi'],
                is_smog_emergency=last_good['is_emergency'],
                **risk_fb,
                **{k: v for k, v in last_good['data'].items() if k != 'hourly_forecast'}
            )
        raise HTTPException(status_code=503, detail="Service not available, try again later!")

@app.post("/rag-query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """RAG-powered Q&A endpoint with Redis caching and graceful degradation"""
    import time
    import logging
    t0 = time.time()
    try:
        # --- Step 1: Check Redis cache for FAQ hit ---
        cached_response = redis_cache.get_cached_response(request.question, request.city or "")
        t1 = time.time(); open('/tmp/rag_timing.log', 'a').write(f"TIMING: Redis cache took {t1 - t0:.3f}s\n")
        if cached_response:
            return RAGQueryResponse(
                answer=cached_response['answer'],
                sources=cached_response.get('sources', []),
                source_ids=cached_response.get('source_ids', []),
                similarity_scores=cached_response.get('similarity_scores', []),
                timestamp=datetime.now().isoformat(),
                current_data=cached_response.get('current_data')
            )

        # --- Step 2: Gather real-time city data ---
        current_data = None
        all_cities_data = []
        
        question_lower = request.question.lower()
        mentioned_cities = []
        for city in PAKISTAN_CITIES.keys():
            if city.lower() in question_lower or (city == "Rawalpindi" and "pindi" in question_lower):
                if city not in mentioned_cities:
                    mentioned_cities.append(city)
        
        if not mentioned_cities and request.city and request.city in PAKISTAN_CITIES:
            mentioned_cities.append(request.city)
            
        for city in mentioned_cities:
            try:
                coords = PAKISTAN_CITIES[city]
                air_data, weather_data, air_forecast, weather_forecast = await asyncio.gather(
                    fetch_air_quality_data(coords['lat'], coords['lon']),
                    fetch_weather_data(coords['lat'], coords['lon']),
                    fetch_air_quality_forecast(coords['lat'], coords['lon']),
                    fetch_weather_forecast(coords['lat'], coords['lon'])
                )
                parsed = parse_responses(air_data, weather_data, air_forecast, weather_forecast)
                parsed['city'] = city
                parsed['aqi'] = calculate_aqi_from_pm25(parsed['pm25'])
                all_cities_data.append(parsed)
                if city == request.city:
                    current_data = parsed
            except:
                pass

        t2 = time.time(); open('/tmp/rag_timing.log', 'a').write(f"TIMING: OpenWeather data gather took {t2 - t1:.3f}s\n")

        # --- Step 3: Execute RAG with graceful degradation ---
        result = await degradation_service.execute_with_fallback(
            rag_system=rag_system,
            query=request.question,
            city=request.city or "Islamabad",
            top_k=request.top_k,
            current_data=current_data,
            all_cities_data=all_cities_data,
            language=request.language
        )
        t3 = time.time(); open('/tmp/rag_timing.log', 'a').write(f"TIMING: RAG execution took {t3 - t2:.3f}s\n")

        # --- Step 4: Cache successful response in Redis ---
        response_to_cache = {
            'answer': result['answer'],
            'sources': result.get('sources', []),
            'source_ids': result.get('source_ids', []),
            'similarity_scores': result.get('similarity_scores', []),
            'current_data': {k: v for k, v in (current_data or {}).items() 
                           if k != 'hourly_forecast'} if current_data else None
        }
        if result.get('tier', 4) <= 2:
            redis_cache.cache_response(request.question, request.city or "", response_to_cache)
        t4 = time.time(); open('/tmp/rag_timing.log', 'a').write(f"TIMING: Total request took {t4 - t0:.3f}s\n")

        return RAGQueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            source_ids=result.get('source_ids', []),
            similarity_scores=result.get('similarity_scores', []),
            timestamp=datetime.now().isoformat(),
            current_data=current_data
        )

    except HTTPException:
        raise
    except Exception as e:
        # Final fallback — return service unavailable
        print(f"⚠️ RAG query completely failed: {e}")
        return RAGQueryResponse(
            answer=degradation_service.service_unavailable_message(),
            sources=[],
            source_ids=[],
            similarity_scores=[],
            timestamp=datetime.now().isoformat(),
            current_data=None
        )

@app.get("/smog-forecast", response_model=ForecastResponse)
async def forecast_smog(
    city: str = Query("Islamabad"),
    lat: Optional[float] = None,
    lon: Optional[float] = None
):
    """48-hour smog forecast with graceful degradation"""
    try:
        if lat is None or lon is None:
            if city not in PAKISTAN_CITIES:
                raise HTTPException(status_code=404, detail=f"City not found")
            coords = PAKISTAN_CITIES[city]
            lat, lon = coords['lat'], coords['lon']

        air_data, weather_data, air_forecast, weather_forecast = await asyncio.gather(
            fetch_air_quality_data(lat, lon),
            fetch_weather_data(lat, lon),
            fetch_air_quality_forecast(lat, lon),
            fetch_weather_forecast(lat, lon)
        )
        parsed = parse_responses(air_data, weather_data, air_forecast, weather_forecast)

        hourly = parsed['hourly_forecast']
        weather_forecast_list = []
        for i in range(48):
            weather_forecast_list.append({
                'temperature': hourly['temperature'][i] if i < len(hourly['temperature']) else 20,
                'humidity': hourly['humidity'][i] if i < len(hourly['humidity']) else 60,
                'wind_speed': hourly['wind_speed'][i] if i < len(hourly['wind_speed']) else 2,
                'pressure': hourly['pressure'][i] if i < len(hourly['pressure']) else 1013,
                'wind_direction': hourly['wind_direction'][i] if i < len(hourly['wind_direction']) else 180
            })

        predictions = ml_pipeline.predict_48h(parsed, weather_forecast_list)

        smog_hours = sum(1 for p in predictions if p['smog_likely'])
        peak_pm25 = max(p['predicted_pm25'] for p in predictions)
        peak_aqi = max(p['predicted_aqi'] for p in predictions)
        peak_hour = next(p['hour'] for p in predictions if p['predicted_pm25'] == peak_pm25)
        avg_confidence = np.mean([p['confidence'] for p in predictions])

        if smog_hours > 36:
            summary = f"SEVERE SMOG EXPECTED for {smog_hours}/48 hours. Peak PM2.5: {peak_pm25:.0f} at hour {peak_hour}. Stay indoors."
        elif smog_hours > 24:
            summary = f"Prolonged smog conditions. {smog_hours} hours affected. Peak PM2.5: {peak_pm25:.0f}."
        elif smog_hours > 12:
            summary = f"Intermittent smog over 48 hours. {smog_hours} affected hours."
        else:
            summary = f"Generally acceptable air quality. {smog_hours} hours with elevated PM2.5."

        return ForecastResponse(
            predictions=[ForecastPrediction(**p) for p in predictions],
            summary=summary,
            smog_hours=smog_hours,
            peak_pm25=peak_pm25,
            peak_aqi=peak_aqi,
            peak_hour=peak_hour,
            average_confidence=avg_confidence
        )

    except HTTPException as he:
        if he.status_code == 404:
            raise
        # Graceful degradation for forecast
        raise HTTPException(status_code=503, detail="Service not available, try again later!")
    except Exception as e:
        print(f"⚠️ Forecast failed for {city}: {e}")
        raise HTTPException(status_code=503, detail="Service not available, try again later!")

@app.post("/predict_pm25", response_model=PredictionResponse)
async def predict_pm25_endpoint(data: PredictionInput):
    """Predict PM2.5 from input pollutant and weather data"""
    try:
        if data.pm10 < 0 or data.pm10 > 1000:
            raise HTTPException(status_code=400, detail="PM10 must be between 0 and 1000 µg/m³")
        if data.no2 < 0 or data.no2 > 500:
            raise HTTPException(status_code=400, detail="NO2 must be between 0 and 500 µg/m³")
        if data.o3 < 0 or data.o3 > 500:
            raise HTTPException(status_code=400, detail="O3 must be between 0 and 500 µg/m³")
        if data.co < 0 or data.co > 50000:
            raise HTTPException(status_code=400, detail="CO must be between 0 and 50000 µg/m³")
        if data.so2 < 0 or data.so2 > 500:
            raise HTTPException(status_code=400, detail="SO2 must be between 0 and 500 µg/m³")
        if data.temperature < -50 or data.temperature > 60:
            raise HTTPException(status_code=400, detail="Temperature must be between -50 and 60 °C")
        if data.relative_humidity < 0 or data.relative_humidity > 100:
            raise HTTPException(status_code=400, detail="Relative humidity must be between 0 and 100 %")

        input_data = data.dict()

        if pm25_prediction_model is None or feature_scaler is None:
            raise HTTPException(
                status_code=503,
                detail="PM2.5 prediction model or feature scaler not available."
            )

        try:
            prediction = predict_pm25_value(input_data)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

        aqi = calculate_aqi_from_pm25(prediction)
        category, message = get_aqi_category_and_message(aqi)

        return PredictionResponse(
            pm25_prediction=round(prediction, 2),
            aqi=round(aqi, 2),
            health_category=category,
            health_message=message,
            input_data=input_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/knowledge-base")
async def get_knowledge_base():
    """Get all documents in knowledge base"""
    return {
        "total_documents": len(KNOWLEDGE_BASE),
        "categories": list(set(doc['category'] for doc in KNOWLEDGE_BASE)),
        "documents": [
            {
                'id': doc['id'],
                'title': doc['title'],
                'category': doc['category'],
                'keywords': doc['keywords']
            }
            for doc in KNOWLEDGE_BASE
        ]
    }

@app.get("/knowledge-base/{doc_id}")
async def get_document(doc_id: str):
    """Get specific document by ID"""
    doc = next((d for d in KNOWLEDGE_BASE if d['id'] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.get("/weaviate-stats")
async def get_weaviate_stats():
    """Get Weaviate collection statistics"""
    if not rag_system.is_initialized:
        return {"status": "not_initialized", "message": "Weaviate RAG system not initialized"}

    try:
        collection = rag_system.weaviate_client.collections.get(rag_system.collection_name)
        response = collection.aggregate.over_all(total_count=True)

        return {
            "status": "initialized",
            "weaviate_url": WEAVIATE_URL,
            "collection_name": rag_system.collection_name,
            "embedding_model": rag_system.model_name,
            "total_documents": response.total_count,
            "categories": list(set(doc['category'] for doc in KNOWLEDGE_BASE)),
            "llm_model": "llama-3.1-8b-instant"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get Weaviate stats: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """System health check — includes Redis status"""
    weaviate_status = "healthy" if rag_system.is_initialized else "not_initialized"

    return {
        "status": "healthy",
        "ml_models": {
            "pm25_forecasting_model": ml_pipeline.pm25_model is not None,
            "smog_classifier": ml_pipeline.smog_classifier is not None,
            "is_trained": ml_pipeline.is_trained,
            "pm25_prediction_model": pm25_prediction_model is not None,
            "feature_scaler": feature_scaler is not None,
            "prediction_ready": (pm25_prediction_model is not None and feature_scaler is not None)
        },
        "rag_system": {
            "status": weaviate_status,
            "vector_db": "Weaviate",
            "embedding_model": rag_system.model_name if rag_system.is_initialized else None,
            "documents": len(rag_system.knowledge_base),
            "weaviate_url": WEAVIATE_URL
        },
        "redis_cache": {
            "status": "connected" if redis_cache.is_connected else "disconnected",
            "url": redis_cache.redis_url,
            "ttl_seconds": redis_cache.default_ttl
        },
        "data": {
            "historical_records": len(historical_data),
            "cache_size": len(cache)
        },
        "groq_configured": bool(GROQ_API_KEY and len(GROQ_API_KEY) > 20),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║ Urban Air Quality Sentinel - Weaviate RAG Backend v3.0       ║
    ║                                                              ║
    ║  ✓ Real-time PM2.5 from OpenWeatherMap API                   ║
    ║  ✓ ML-powered 48-hour forecasting                            ║
    ║  ✓ RAG System: Weaviate + Sentence Transformers + Groq       ║
    ║  ✓ PM2.5 Prediction from pollutant data (Random Forest)      ║
    ║  ✓ Multi-city coverage (10 Pakistani cities)                 ║
    ╚══════════════════════════════════════════════════════════════╝

    📚 Weaviate RAG System:
       • Vector Store: Weaviate
       • Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
       • LLM: Groq LLaMA 3.3 70B
       • Knowledge Base: Comprehensive air quality documents

    🔧 Weaviate Setup (Local):
       docker run -d -p 8080:8080 \\
       -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\
       -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \\
       semitechnologies/weaviate:latest

    🚀 Starting server on http://localhost:8000
    📖 API Documentation: http://localhost:8000/docs
    📖 Health Check: http://localhost:8000/health
    🧠 Weaviate Stats: http://localhost:8000/weaviate-stats

    ⚠️ Requirements:
       pip install fastapi uvicorn httpx numpy pandas scikit-learn
       pip install sentence-transformers weaviate-client joblib

    🔑 Environment Variables:
       • WEAVIATE_URL: http://localhost:8080 (default)
       • WEAVIATE_API_KEY: Optional for Weaviate Cloud
       • GROQ_API_KEY: Required for LLM generation
    """)

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )