import React, { useState, useEffect, useCallback } from 'react';
import { Cloud, AlertTriangle, TrendingUp, Wind, Thermometer, MapPin, Activity, Zap, Shield, Flame, Target, Layers, MessageSquare, Send, BookOpen, BarChart3, Brain, Sparkles, TrendingDown, AlertCircle, CheckCircle, XCircle, ArrowUp, ArrowDown, Minus, RefreshCw, Gauge, Eye, Droplets, Calculator } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter, ZAxis, ComposedChart, Cell } from 'recharts';

const GROQ_API_KEY = '';
const BACKEND_URL = 'http://localhost:8000';

const KNOWLEDGE_BASE = [
  {
    id: 'pm25_basics',
    category: 'pollutants',
    title: 'What is PM2.5?',
    content: 'PM2.5 refers to fine particulate matter with a diameter of 2.5 micrometers or less. These particles are small enough to penetrate deep into the lungs and even enter the bloodstream, causing serious health issues including respiratory diseases, cardiovascular problems, and premature death. Common sources include vehicle emissions, industrial activities, construction dust, and biomass burning.',
    keywords: ['pm2.5', 'particulate matter', 'fine particles', 'pollution', 'what is']
  },
  {
    id: 'smog_formation',
    category: 'science',
    title: 'How Smog Forms',
    content: 'Smog forms when pollutants like PM2.5, nitrogen oxides, and volatile organic compounds accumulate in the atmosphere under specific weather conditions. Key factors include: temperature inversion (cold air trapped under warm air), low wind speeds (less than 2 m/s), high humidity (above 70%), high atmospheric pressure, and increased emissions from traffic and industry. Winter months in Pakistan (November-February) are particularly prone to severe smog events.',
    keywords: ['smog', 'formation', 'how', 'why', 'causes', 'temperature inversion']
  },
  {
    id: 'health_impacts',
    category: 'health',
    title: 'Health Effects of Air Pollution',
    content: 'Short-term exposure to high PM2.5 levels causes eye irritation, coughing, breathing difficulties, and asthma attacks. Long-term exposure increases risks of chronic bronchitis, reduced lung function, heart disease, stroke, and lung cancer. Children, elderly, pregnant women, and people with pre-existing conditions are most vulnerable. At PM2.5 levels above 250 µg/m³, everyone experiences severe health effects and outdoor activities should be completely avoided.',
    keywords: ['health', 'effects', 'impacts', 'symptoms', 'risks', 'diseases']
  },
  {
    id: 'protection_measures',
    category: 'safety',
    title: 'How to Protect Yourself',
    content: 'During high pollution: Stay indoors with windows closed. Use N95 or higher-rated masks if going outside. Run air purifiers with HEPA filters indoors. Avoid outdoor exercise, especially during peak traffic hours (7-9 AM, 5-8 PM). Keep children and elderly indoors. Use public transport or carpool to reduce emissions. Monitor air quality apps regularly. Seek medical attention if experiencing severe symptoms like chest pain or extreme breathlessness.',
    keywords: ['protection', 'safety', 'masks', 'air purifiers', 'how to protect', 'precautions']
  }
];

class SimpleRAGSystem {
  constructor(knowledgeBase) {
    this.documents = knowledgeBase;
  }
  
  calculateSimilarity(query, document) {
    const queryTokens = query.toLowerCase().split(/\s+/);
    const docText = (document.content + ' ' + document.title + ' ' + document.keywords.join(' ')).toLowerCase();
    
    let score = 0;
    queryTokens.forEach(token => {
      if (document.keywords.some(kw => kw.includes(token))) score += 3;
      if (document.title.toLowerCase().includes(token)) score += 2;
      const matches = (docText.match(new RegExp(token, 'g')) || []).length;
      score += matches * 0.5;
    });
    
    return score;
  }
  
  retrieveRelevant(query, topK = 3) {
    const scored = this.documents.map(doc => ({
      ...doc,
      score: this.calculateSimilarity(query, doc)
    }));
    
    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter(doc => doc.score > 0);
  }
  
  async generateAnswer(query, context, currentData = null) {
    const systemPrompt = `You are an expert environmental scientist specializing in air quality and public health in Pakistan. Use the provided context to answer questions accurately and concisely.`;
    
    let userPrompt = `Context from knowledge base:\n\n`;
    context.forEach((doc, idx) => {
      userPrompt += `[Source ${idx + 1}: ${doc.title}]\n${doc.content}\n\n`;
    });
    
    if (currentData) {
      userPrompt += `\nCurrent Real-time Data for ${currentData.city ?? 'Unknown'}:\n`;
      userPrompt += `- PM2.5: ${(currentData.pm25 ?? 0).toFixed(1)} µg/m³\n`;
      userPrompt += `- AQI: ${(currentData.aqi ?? 0).toFixed(0)}\n`;
      userPrompt += `- Temperature: ${(currentData.temperature ?? 0).toFixed(1)}°C\n`;
      userPrompt += `- Humidity: ${(currentData.humidity ?? 0).toFixed(0)}%\n`;
      userPrompt += `- Wind Speed: ${(currentData.wind_speed ?? 0).toFixed(1)} m/s\n\n`;
    }
    
    userPrompt += `Question: ${query}\n\nProvide a clear, concise answer (3-5 sentences).`;
    
    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${GROQ_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'llama-3.3-70b-versatile',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ],
          temperature: 0.7,
          max_tokens: 500
        })
      });
      
      if (!response.ok) throw new Error('API request failed');
      
      const data = await response.json();
      return {
        answer: data.choices[0].message.content,
        sources: context.map(doc => doc.title),
        success: true
      };
    } catch (error) {
      return {
        answer: 'I apologize, but I encountered an error. Please try again.',
        sources: [],
        success: false
      };
    }
  }
}

const ragSystem = new SimpleRAGSystem(KNOWLEDGE_BASE);

class SmogDetectionSystem {
  detectSmog(currentData) {
    const pm25 = currentData?.pm25 ?? 0;
    const wind_speed = currentData?.wind_speed ?? 0;
    const humidity = currentData?.humidity ?? 0;
    const pressure = currentData?.pressure ?? 0;
    const temperature = currentData?.temperature ?? 0;
    let smogScore = 0;
    let factors = [];
    
    if (pm25 > 250) {
      smogScore += 40;
      factors.push('Extreme PM2.5 levels');
    } else if (pm25 > 150) {
      smogScore += 30;
      factors.push('Very high PM2.5');
    } else if (pm25 > 100) {
      smogScore += 20;
      factors.push('Elevated PM2.5');
    }
    
    if (wind_speed < 1.5) {
      smogScore += 25;
      factors.push('Stagnant air (wind < 1.5 m/s)');
    } else if (wind_speed < 2.5) {
      smogScore += 15;
      factors.push('Low wind speed');
    }
    
    if (humidity > 80) {
      smogScore += 20;
      factors.push('Very high humidity');
    } else if (humidity > 70) {
      smogScore += 10;
      factors.push('High humidity');
    }
    
    if (pressure > 1020) {
      smogScore += 10;
      factors.push('High atmospheric pressure');
    }
    
    if (temperature < 15) {
      smogScore += 5;
      factors.push('Cold temperature');
    }
    
    let severity, color, actions;
    if (smogScore >= 80) {
      severity = 'EXTREME';
      color = '#7f1d1d';
      actions = ['Stay indoors at all times', 'Use air purifiers continuously', 'Wear N95 masks if must go outside'];
    } else if (smogScore >= 60) {
      severity = 'SEVERE';
      color = '#991b1b';
      actions = ['Minimize outdoor activities', 'Use air purifiers', 'Wear N95 masks outdoors'];
    } else if (smogScore >= 40) {
      severity = 'HIGH';
      color = '#dc2626';
      actions = ['Limit outdoor exposure', 'Sensitive groups stay indoors', 'Consider wearing masks'];
    } else if (smogScore >= 20) {
      severity = 'MODERATE';
      color = '#f97316';
      actions = ['Sensitive individuals limit prolonged outdoor activities', 'Monitor air quality'];
    } else {
      severity = 'LOW';
      color = '#10b981';
      actions = ['Air quality is acceptable', 'Normal activities can continue'];
    }
    
    return {
      isSmog: smogScore >= 40,
      severity,
      probability: Math.min(smogScore, 100),
      score: smogScore,
      factors,
      color,
      actions
    };
  }
}

const smogDetector = new SmogDetectionSystem();

class AIInsightsEngine {
  async analyzeData(currentData, historicalData, forecast) {
    const insights = {
      trends: this.detectTrends(historicalData),
      anomalies: this.detectAnomalies(currentData),
      correlations: this.findCorrelations(currentData),
      recommendations: this.generateRecommendations(currentData, forecast)
    };
    
    const aiInterpretation = await this.generateAIInterpretation(currentData, insights);
    return { ...insights, aiInterpretation };
  }
  
  detectTrends(historicalData) {
    if (!historicalData || historicalData.length < 2) return null;
    
    const latest = historicalData[historicalData.length - 1];
    const previous = historicalData[0];
    
    const pm25Change = ((latest.pm25 - previous.pm25) / previous.pm25) * 100;
    const aqiChange = ((latest.aqi - previous.aqi) / previous.aqi) * 100;
    
    return {
      pm25Trend: pm25Change > 10 ? 'increasing' : pm25Change < -10 ? 'decreasing' : 'stable',
      pm25Change: pm25Change.toFixed(1),
      aqiTrend: aqiChange > 10 ? 'worsening' : aqiChange < -10 ? 'improving' : 'stable',
      aqiChange: aqiChange.toFixed(1),
      direction: pm25Change > 0 ? 'up' : pm25Change < 0 ? 'down' : 'stable'
    };
  }
  
  detectAnomalies(currentData) {
    const anomalies = [];
    
    const pm25 = currentData?.pm25 ?? 0;
    const wind_speed = currentData?.wind_speed ?? 0;
    const humidity = currentData?.humidity ?? 0;
    
    if (pm25 > 300) {
      anomalies.push({
        type: 'critical',
        metric: 'PM2.5',
        value: pm25.toFixed(1),
        message: 'PM2.5 at hazardous levels - immediate action required'
      });
    }
    
    if (wind_speed < 1.0) {
      anomalies.push({
        type: 'warning',
        metric: 'Wind Speed',
        value: wind_speed.toFixed(1),
        message: 'Extremely low wind causing pollutant accumulation'
      });
    }
    
    if (humidity > 85) {
      anomalies.push({
        type: 'warning',
        metric: 'Humidity',
        value: humidity.toFixed(0),
        message: 'Very high humidity may worsen visibility and air quality'
      });
    }
    
    return anomalies;
  }
  
  findCorrelations(currentData) {
    const correlations = [];
    
    const pm25 = currentData?.pm25 ?? 0;
    const wind_speed = currentData?.wind_speed ?? 0;
    const humidity = currentData?.humidity ?? 0;
    const temperature = currentData?.temperature ?? 0;
    const pressure = currentData?.pressure ?? 0;
    
    if (wind_speed < 2 && pm25 > 150) {
      correlations.push({
        factor1: 'Low Wind Speed',
        factor2: 'High PM2.5',
        strength: 'Strong',
        explanation: 'Stagnant air prevents pollutant dispersion'
      });
    }
    
    if (humidity > 75) {
      correlations.push({
        factor1: 'High Humidity',
        factor2: 'Reduced Visibility',
        strength: 'Moderate',
        explanation: 'Moisture causes particles to grow, reducing visibility'
      });
    }
    
    if (temperature < 15 && pressure > 1020) {
      correlations.push({
        factor1: 'Cold + High Pressure',
        factor2: 'Poor Dispersion',
        strength: 'Strong',
        explanation: 'Temperature inversion traps pollutants near ground'
      });
    }
    
    return correlations;
  }
  
  generateRecommendations(currentData, forecast) {
    const recommendations = [];
    
    const aqi = currentData?.aqi ?? 0;
    const wind_speed = currentData?.wind_speed ?? 0;
    
    if (aqi > 200) {
      recommendations.push({
        priority: 'critical',
        action: 'Stay Indoors',
        reason: 'AQI in very unhealthy range',
        duration: 'Until AQI drops below 150'
      });
    }
    
    if (wind_speed < 2) {
      recommendations.push({
        priority: 'high',
        action: 'Use Air Purifiers',
        reason: 'Poor natural ventilation',
        duration: 'Continuously'
      });
    }
    
    if (forecast && forecast.smog_hours > 24) {
      recommendations.push({
        priority: 'high',
        action: 'Stock N95 Masks',
        reason: 'Extended smog period expected',
        duration: 'Next 48 hours'
      });
    }
    
    return recommendations;
  }
  
  async generateAIInterpretation(currentData, insights) {
    const prompt = `Based on current air quality data in ${currentData?.city ?? 'Unknown'}:
- PM2.5: ${(currentData?.pm25 ?? 0).toFixed(1)} µg/m³
- AQI: ${(currentData?.aqi ?? 0).toFixed(0)}
- Wind Speed: ${(currentData?.wind_speed ?? 0).toFixed(1)} m/s
- Humidity: ${(currentData?.humidity ?? 0).toFixed(0)}%
- Temperature: ${(currentData?.temperature ?? 0).toFixed(1)}°C
${insights.trends ? `Trend: PM2.5 ${insights.trends.pm25Trend} by ${insights.trends.pm25Change}%` : ''}

Provide a brief 2-3 sentence expert analysis explaining WHY pollution is at current levels, focusing on meteorological and emission factors.`;
    
    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${GROQ_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'llama-3.3-70b-versatile',
          messages: [
            { role: 'system', content: 'You are an expert environmental scientist. Provide concise, scientific explanations about air quality.' },
            { role: 'user', content: prompt }
          ],
          temperature: 0.7,
          max_tokens: 200
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        return data.choices[0].message.content;
      }
    } catch (error) {
      console.error('AI interpretation error:', error);
    }
    
    return 'Current pollution levels are influenced by meteorological conditions and local emissions. Monitor updates for changes.';
  }
}

const insightsEngine = new AIInsightsEngine();

export default function SmogSentinelTabs() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [allCitiesData, setAllCitiesData] = useState({});
  const [currentData, setCurrentData] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [smogDetection, setSmogDetection] = useState(null);
  const [aiInsights, setAIInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCity, setSelectedCity] = useState('Lahore');
  const [lastUpdate, setLastUpdate] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [chatMessages, setChatMessages] = useState([]);
  const [userQuestion, setUserQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  
  const [predictionInputs, setPredictionInputs] = useState({
    pm10: '',
    no2: '',
    o3: '',
    co: '',
    so2: '',
    temperature: '',
    relative_humidity: ''
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  
  const cities = [
    { name: 'Lahore', lat: 31.5204, lon: 74.3587, color: '#6366f1' },
    { name: 'Rawalpindi', lat: 33.5651, lon: 73.0169, color: '#8b5cf6' },
    { name: 'Islamabad', lat: 33.6844, lon: 73.0479, color: '#10b981' },
    { name: 'Karachi', lat: 24.8607, lon: 67.0011, color: '#3b82f6' },
    { name: 'Faisalabad', lat: 31.4504, lon: 73.1350, color: '#f59e0b' },
    { name: 'Multan', lat: 30.1575, lon: 71.5249, color: '#ec4899' }
  ];
  
  const suggestedQuestions = [
    "What is causing the smog right now?",
    "Is it safe to go outside today?",
    "How can I protect my family from smog?",
    "Why is smog worse in winter?"
  ];
  
  const fetchCityData = useCallback(async (cityName) => {
    try {
      const response = await fetch(`${BACKEND_URL}/current-air-quality?city=${encodeURIComponent(cityName)}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (err) {
      console.error(`Error fetching data for ${cityName}:`, err);
      return null;
    }
  }, []);
  
  const fetchForecast = useCallback(async (cityName) => {
    try {
      const response = await fetch(`${BACKEND_URL}/smog-forecast?city=${encodeURIComponent(cityName)}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (err) {
      console.error(`Error fetching forecast for ${cityName}:`, err);
      return null;
    }
  }, []);
  
  const fetchAllCitiesData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const citiesDataPromises = cities.map(city => fetchCityData(city.name));
      const citiesResults = await Promise.all(citiesDataPromises);
      
      const citiesDataMap = {};
      citiesResults.forEach((data, index) => {
        if (data) {
          const city = cities[index];
          citiesDataMap[city.name] = {
            ...data,
            city: city.name,
            lat: city.lat,
            lon: city.lon,
            color: city.color
          };
        }
      });
      
      setAllCitiesData(citiesDataMap);
      
      if (citiesDataMap[selectedCity]) {
        const selectedCityData = citiesDataMap[selectedCity];
        setCurrentData(selectedCityData);
        
        const smog = smogDetector.detectSmog(selectedCityData);
        setSmogDetection(smog);
        
        const forecastData = await fetchForecast(selectedCity);
        setForecast(forecastData);
        
        setHistoricalData(prev => {
          const newHistory = [...prev, selectedCityData];
          return newHistory.slice(-10);
        });
        
        const insights = await insightsEngine.analyzeData(
          selectedCityData,
          historicalData.length > 0 ? historicalData : [selectedCityData],
          forecastData
        );
        setAIInsights(insights);
      }
      
      setLastUpdate(new Date());
      setLoading(false);
    } catch (err) {
      console.error('Fetch error:', err);
      setError(err.message);
      setLoading(false);
    }
  }, [selectedCity, fetchCityData, fetchForecast, historicalData]);
  
  useEffect(() => {
    fetchAllCitiesData();
    //const interval = setInterval(fetchAllCitiesData, 300000);
    //return () => clearInterval(interval);
  }, []);
  
  useEffect(() => {
    if (allCitiesData[selectedCity]) {
      const selectedCityData = allCitiesData[selectedCity];
      setCurrentData(selectedCityData);
      
      const smog = smogDetector.detectSmog(selectedCityData);
      setSmogDetection(smog);
      
      fetchForecast(selectedCity).then(forecastData => {
        setForecast(forecastData);
      });
    }
  }, [selectedCity, allCitiesData, fetchForecast]);
  
  const handlePredictionInputChange = (e) => {
    const { name, value } = e.target;
    setPredictionInputs(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleUseCurrentData = () => {
    if (currentData) {
      setPredictionInputs({
        pm10: (currentData.pm10 || 0).toFixed(2),
        no2: (currentData.no2 || 0).toFixed(2),
        o3: (currentData.o3 || 0).toFixed(2),
        co: (currentData.co || 0).toFixed(2),
        so2: (currentData.so2 || 0).toFixed(2),
        temperature: (currentData.temperature || 0).toFixed(2),
        relative_humidity: (currentData.humidity || 0).toFixed(2)
      });
    }
  };
  
  const handlePredict = async (e) => {
    e.preventDefault();
    setPredictionLoading(true);
    setPredictionError(null);
    setPredictionResult(null);
    
    try {
      const payload = {};
      for (const key in predictionInputs) {
        const val = parseFloat(predictionInputs[key]);
        if (isNaN(val)) {
          throw new Error(`Invalid input for ${key}. Please enter a valid number.`);
        }
        payload[key] = val;
      }
      
      console.log('Sending prediction request:', payload);
      
      const response = await fetch(`${BACKEND_URL}/predict_pm25`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction failed');
      }
      
      const data = await response.json();
      console.log('Prediction response:', data);
      
      // Calculate AQI from PM2.5 if not provided
      const calculateAQI = (pm25) => {
        if (pm25 <= 12) return (50 / 12) * pm25;
        if (pm25 <= 35.4) return 50 + ((100 - 50) / (35.4 - 12)) * (pm25 - 12);
        if (pm25 <= 55.4) return 100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4);
        if (pm25 <= 150.4) return 150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4);
        if (pm25 <= 250.4) return 200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4);
        return Math.min(300 + ((500 - 300) / (500 - 250.4)) * (pm25 - 250.4), 500);
      };
      
      const getHealthInfo = (aqi) => {
        if (aqi <= 50) return { category: 'Good', message: 'Air quality is satisfactory, and air pollution poses little or no risk.' };
        if (aqi <= 100) return { category: 'Moderate', message: 'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.' };
        if (aqi <= 150) return { category: 'Unhealthy for Sensitive Groups', message: 'Members of sensitive groups may experience health effects. The general public is less likely to be affected.' };
        if (aqi <= 200) return { category: 'Unhealthy', message: 'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.' };
        if (aqi <= 300) return { category: 'Very Unhealthy', message: 'Health alert: The risk of health effects is increased for everyone. Sensitive groups should avoid outdoor activities.' };
        return { category: 'Hazardous', message: 'Health warning of emergency conditions: everyone is more likely to be affected. Stay indoors and avoid all outdoor activities.' };
      };
      
      const pm25Value = data.pm25_prediction || 0;
      const aqiValue = calculateAQI(pm25Value);
      const healthInfo = getHealthInfo(aqiValue);
      
      setPredictionResult({
        pm25: pm25Value,
        aqi: aqiValue,
        health_category: healthInfo.category,
        health_message: healthInfo.message,
        input_data: payload
      });
    } catch (err) {
      console.error('Prediction error:', err);
      setPredictionError(err.message);
    } finally {
      setPredictionLoading(false);
    }
  };
  
  const handleAskQuestion = async (question) => {
    if (!question.trim()) return;
    
    setIsAsking(true);
    const userMsg = {
      role: 'user',
      content: question,
      timestamp: new Date()
    };
    setChatMessages(prev => [...prev, userMsg]);
    setUserQuestion('');
    
    const relevantDocs = ragSystem.retrieveRelevant(question, 3);
    const contextData = currentData ? { ...currentData, smogDetection } : null;
    const result = await ragSystem.generateAnswer(question, relevantDocs, contextData);
    
    const assistantMsg = {
      role: 'assistant',
      content: result.answer,
      sources: result.sources,
      timestamp: new Date()
    };
    setChatMessages(prev => [...prev, assistantMsg]);
    setIsAsking(false);
  };
  
  const getAQIColor = (aqi) => {
    if (aqi <= 50) return '#10b981';
    if (aqi <= 100) return '#fbbf24';
    if (aqi <= 150) return '#f97316';
    if (aqi <= 200) return '#ef4444';
    if (aqi <= 300) return '#a855f7';
    return '#dc2626';
  };
  
  const getAQILabel = (aqi) => {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Moderate';
    if (aqi <= 150) return 'Unhealthy for Sensitive';
    if (aqi <= 200) return 'Unhealthy';
    if (aqi <= 300) return 'Very Unhealthy';
    return 'Hazardous';
  };
  
  if (loading && !currentData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-indigo-500 mx-auto"></div>
            <Cloud className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-indigo-500" />
          </div>
          <p className="mt-4 text-indigo-600 font-medium">Loading Real-Time Data...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-red-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl p-6 max-w-md shadow-lg border border-red-200">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-red-700 mb-2 text-center">Connection Error</h3>
          <p className="text-red-600 text-sm mb-4 text-center">{error}</p>
          <button
            onClick={fetchAllCitiesData}
            className="w-full bg-indigo-500 hover:bg-indigo-600 text-white py-2 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Retry Connection
          </button>
        </div>
      </div>
    );
  }
  
  const radarData = currentData ? [
    { metric: 'PM2.5', value: Math.min(((currentData?.pm25 ?? 0) / 300) * 100, 100) },
    { metric: 'PM10', value: Math.min(((currentData?.pm10 ?? 0) / 600) * 100, 100) },
    { metric: 'NO₂', value: Math.min(((currentData?.no2 ?? 0) / 200) * 100, 100) },
    { metric: 'O₃', value: Math.min(((currentData?.o3 ?? 0) / 180) * 100, 100) },
    { metric: 'Humidity', value: currentData?.humidity ?? 0 },
    { metric: 'Wind', value: Math.min(((currentData?.wind_speed ?? 0) / 10) * 100, 100) }
  ] : [];
  
  const mapData = Object.values(allCitiesData).map(city => ({
    city: city.city,
    lat: city.lat,
    lon: city.lon,
    pm25: city.pm25 ?? 0,
    aqi: city.aqi ?? 0,
    color: getAQIColor(city.aqi ?? 0),
    size: Math.min(Math.max((city.pm25 ?? 0) / 10, 10), 100)
  }));
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-6 max-w-7xl">
        <header className="mb-6">
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center justify-between mb-4 flex-wrap gap-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-xl flex items-center justify-center">
                  <Shield className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-800">Real-Time Smog Detection System</h1>
                  <p className="text-sm text-gray-500">Live monitoring with AI-powered insights • Updates every 5 minutes</p>
                </div>
              </div>
              <button
                onClick={fetchAllCitiesData}
                disabled={loading}
                className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center gap-2 text-sm"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
            
            <div className="flex items-center gap-3 flex-wrap">
              <select
                value={selectedCity}
                onChange={(e) => setSelectedCity(e.target.value)}
                className="px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:border-indigo-400 text-sm text-gray-700"
              >
                {cities.map(city => (
                  <option key={city.name} value={city.name}>📍 {city.name}</option>
                ))}
              </select>
              
              {lastUpdate && (
                <div className="flex items-center gap-2 text-gray-500 text-xs bg-gray-50 px-3 py-2 rounded-lg">
                  <Activity className="w-3 h-3 text-green-500 animate-pulse" />
                  Last updated: {lastUpdate.toLocaleTimeString()}
                </div>
              )}
            </div>
          </div>
        </header>
        
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 mb-6 overflow-hidden">
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex-1 px-6 py-4 font-semibold text-sm transition-all flex items-center justify-center gap-2 ${
                activeTab === 'dashboard' ? 'bg-gradient-to-r from-indigo-500 to-blue-500 text-white' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <BarChart3 className="w-5 h-5" />
              Data Visualization
            </button>
            <button
              onClick={() => setActiveTab('insights')}
              className={`flex-1 px-6 py-4 font-semibold text-sm transition-all flex items-center justify-center gap-2 ${
                activeTab === 'insights' ? 'bg-gradient-to-r from-indigo-500 to-blue-500 text-white' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <Brain className="w-5 h-5" />
              AI Insights
            </button>
            <button
              onClick={() => setActiveTab('assistant')}
              className={`flex-1 px-6 py-4 font-semibold text-sm transition-all flex items-center justify-center gap-2 ${
                activeTab === 'assistant' ? 'bg-gradient-to-r from-indigo-500 to-blue-500 text-white' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <MessageSquare className="w-5 h-5" />
              AI Assistant
            </button>
            <button
              onClick={() => setActiveTab('prediction')}
              className={`flex-1 px-6 py-4 font-semibold text-sm transition-all flex items-center justify-center gap-2 ${
                activeTab === 'prediction' ? 'bg-gradient-to-r from-indigo-500 to-blue-500 text-white' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <Calculator className="w-5 h-5" />
              PM2.5 Prediction
            </button>
          </div>
        </div>
        
        {activeTab === 'dashboard' && currentData && (
          <div className="space-y-6">
            {smogDetection && smogDetection.severity !== 'LOW' && (
              <div className="rounded-3xl shadow-xl border-l-4 p-6 bg-white/80 backdrop-blur-sm" style={{
                borderLeftColor: smogDetection.color
              }}>
                <div className="flex items-start gap-3">
                  <Flame className="w-8 h-8 flex-shrink-0 mt-1" style={{ color: smogDetection.color }} />
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h2 className="text-2xl font-bold" style={{ color: smogDetection.color }}>
                        {smogDetection.severity} SMOG ALERT
                      </h2>
                      <span className="px-3 py-1 rounded-full text-sm font-bold text-white shadow-lg" style={{ backgroundColor: smogDetection.color }}>
                        {smogDetection.probability.toFixed(0)}% Probability
                      </span>
                    </div>
                    <p className="text-gray-700 mb-4 font-medium">
                      Smog conditions detected in {selectedCity}. Risk Score: {smogDetection.score}/100
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-2xl p-4 border border-orange-200">
                        <h3 className="font-semibold text-sm mb-2" style={{ color: smogDetection.color }}>
                          Contributing Factors:
                        </h3>
                        <ul className="space-y-1">
                          {smogDetection.factors.map((factor, idx) => (
                            <li key={idx} className="text-sm text-gray-700 flex items-center gap-2">
                              <Target className="w-3 h-3" style={{ color: smogDetection.color }} />
                              {factor}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-4 border border-blue-200">
                        <h3 className="font-semibold text-sm mb-2" style={{ color: smogDetection.color }}>
                          Recommended Actions:
                        </h3>
                        <ul className="space-y-1">
                          {smogDetection.actions.slice(0, 3).map((action, idx) => (
                            <li key={idx} className="text-sm text-gray-700 flex items-center gap-2">
                              <Shield className="w-3 h-3" style={{ color: smogDetection.color }} />
                              {action}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-indigo-100 p-6">
              <div className="flex items-center gap-3 mb-6">
                <Eye className="w-6 h-6 text-indigo-500" />
                <div>
                  <h2 className="text-xl font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Comprehensive Data Analysis</h2>
                  <p className="text-sm text-gray-500">Multi-dimensional visualization with derived insights</p>
                </div>
              </div>
              
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Regional Air Quality Distribution</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                    <XAxis 
                      type="number" 
                      dataKey="lon" 
                      name="Longitude" 
                      domain={[66, 75]} 
                      tick={{ fill: '#d1d5db', fontSize: 12 }} 
                      stroke="#f3f4f6"
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis 
                      type="number" 
                      dataKey="lat" 
                      name="Latitude" 
                      domain={[23, 35]} 
                      tick={{ fill: '#d1d5db', fontSize: 12 }} 
                      stroke="#f3f4f6"
                      axisLine={false}
                      tickLine={false}
                    />
                    <ZAxis type="number" dataKey="size" range={[400, 1400]} name="PM2.5" />
                    <Tooltip 
                      cursor={false}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-white/98 backdrop-blur-lg p-5 rounded-2xl shadow-2xl border-2" style={{ borderColor: data.color }}>
                              <p className="font-bold text-gray-900 mb-3 text-lg">{data.city}</p>
                              <div className="space-y-2">
                                <div className="flex justify-between items-center gap-6">
                                  <span className="text-sm text-gray-600">PM2.5</span>
                                  <span className="font-bold text-base text-gray-900">{data.pm25.toFixed(1)} µg/m³</span>
                                </div>
                                <div className="flex justify-between items-center gap-6">
                                  <span className="text-sm text-gray-600">AQI</span>
                                  <span className="font-bold text-base" style={{ color: data.color }}>{data.aqi.toFixed(0)}</span>
                                </div>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }} 
                    />
                    <Scatter data={mapData}>
                      {mapData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} opacity={0.85} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-4">Multi-City AQI Comparison</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={Object.values(allCitiesData).map(city => ({
                      city: city.city,
                      aqi: city.aqi ?? 0,
                      pm25: city.pm25 ?? 0,
                      fill: getAQIColor(city.aqi ?? 0)
                    }))}>
                      <XAxis 
                        dataKey="city" 
                        tick={{ fill: '#d1d5db', fontSize: 11 }} 
                        angle={-15} 
                        textAnchor="end" 
                        height={60} 
                        stroke="#f3f4f6"
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis 
                        tick={{ fill: '#d1d5db', fontSize: 12 }} 
                        stroke="#f3f4f6"
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip 
                        cursor={false}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white/98 backdrop-blur-lg p-4 rounded-2xl shadow-2xl border-2" style={{ borderColor: data.fill }}>
                                <p className="font-bold text-gray-900 mb-2 text-base">{data.city}</p>
                                <div className="space-y-1">
                                  <div className="flex justify-between items-center gap-4">
                                    <span className="text-sm text-gray-600">AQI</span>
                                    <span className="font-bold text-base" style={{ color: data.fill }}>{data.aqi.toFixed(0)}</span>
                                  </div>
                                  <div className="flex justify-between items-center gap-4">
                                    <span className="text-sm text-gray-600">PM2.5</span>
                                    <span className="font-bold text-base text-gray-900">{data.pm25.toFixed(1)} µg/m³</span>
                                  </div>
                                </div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar dataKey="aqi" radius={[12, 12, 0, 0]} maxBarSize={60}>
                        {Object.values(allCitiesData).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getAQIColor(entry.aqi ?? 0)} opacity={0.85} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-4">Pollutant Radar</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#e5e7eb" strokeWidth={1} />
                      <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                      <PolarRadiusAxis 
                        domain={[0, 100]} 
                        tick={{ fill: '#d1d5db', fontSize: 10 }} 
                        stroke="#f3f4f6"
                        axisLine={false}
                      />
                      <Tooltip 
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white/98 backdrop-blur-lg p-4 rounded-2xl shadow-2xl border-2 border-indigo-200">
                                <p className="font-bold text-gray-900 mb-2 text-base">{data.metric}</p>
                                <div className="flex justify-between items-center gap-4">
                                  <span className="text-sm text-gray-600">Level</span>
                                  <span className="font-bold text-base text-indigo-600">{data.value.toFixed(1)}%</span>
                                </div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Radar dataKey="value" stroke="#a5b4fc" fill="#c7d2fe" fillOpacity={0.65} strokeWidth={2.5} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            
            <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-indigo-100 p-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">Real-Time Air Quality Metrics - {selectedCity}</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl p-4 border border-indigo-200 shadow-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Gauge className="w-5 h-5 text-indigo-600" />
                    <span className="px-2 py-1 rounded-full text-xs font-semibold text-white shadow-md" style={{ backgroundColor: getAQIColor(currentData?.aqi ?? 0) }}>
                      {getAQILabel(currentData?.aqi ?? 0)}
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 mb-1">AQI</p>
                  <p className="text-3xl font-bold text-indigo-700">{(currentData?.aqi ?? 0).toFixed(0)}</p>
                </div>
                
                <div className="bg-gradient-to-br from-rose-100 to-pink-100 rounded-2xl p-4 border border-rose-200 shadow-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Cloud className="w-5 h-5 text-rose-600" />
                  </div>
                  <p className="text-xs text-gray-600 mb-1">PM2.5</p>
                  <p className="text-3xl font-bold text-rose-700">{(currentData?.pm25 ?? 0).toFixed(1)}</p>
                  <p className="text-xs text-gray-500">µg/m³</p>
                </div>
                
                <div className="bg-gradient-to-br from-cyan-100 to-blue-100 rounded-2xl p-4 border border-cyan-200 shadow-lg">
                  <Wind className="w-5 h-5 text-cyan-600 mb-2" />
                  <p className="text-xs text-gray-600 mb-1">Wind Speed</p>
                  <p className="text-3xl font-bold text-cyan-700">{(currentData?.wind_speed ?? 0).toFixed(1)}</p>
                  <p className="text-xs text-gray-500">m/s</p>
                </div>
                
                <div className="bg-gradient-to-br from-amber-100 to-orange-100 rounded-2xl p-4 border border-amber-200 shadow-lg">
                  <Thermometer className="w-5 h-5 text-amber-600 mb-2" />
                  <p className="text-xs text-gray-600 mb-1">Temperature</p>
                  <p className="text-3xl font-bold text-amber-700">{(currentData?.temperature ?? 0).toFixed(1)}</p>
                  <p className="text-xs text-gray-500">°C</p>
                </div>
              </div>
              
              <div className="mt-4 pt-4 border-t border-indigo-100">
                <h3 className="text-sm font-semibold text-gray-700 mb-3">Additional Pollutants</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-3 border border-gray-200 shadow-sm">
                    <p className="text-xs text-gray-600 mb-1">PM10</p>
                    <p className="text-lg font-bold text-gray-800">{(currentData.pm10 ?? 0).toFixed(1)}</p>
                  </div>
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-3 border border-gray-200 shadow-sm">
                    <p className="text-xs text-gray-600 mb-1">NO₂</p>
                    <p className="text-lg font-bold text-gray-800">{(currentData.no2 ?? 0).toFixed(1)}</p>
                  </div>
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-3 border border-gray-200 shadow-sm">
                    <p className="text-xs text-gray-600 mb-1">O₃</p>
                    <p className="text-lg font-bold text-gray-800">{(currentData.o3 ?? 0).toFixed(1)}</p>
                  </div>
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-3 border border-blue-200 shadow-sm">
                    <Droplets className="w-3 h-3 text-blue-500 mb-1" />
                    <p className="text-xs text-gray-600 mb-1">Humidity</p>
                    <p className="text-lg font-bold text-gray-800">{(currentData.humidity ?? 0).toFixed(0)}%</p>
                  </div>
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-3 border border-gray-200 shadow-sm">
                    <p className="text-xs text-gray-600 mb-1">Pressure</p>
                    <p className="text-lg font-bold text-gray-800">{(currentData.pressure ?? 0).toFixed(0)}</p>
                  </div>
                </div>
              </div>
            </div>
            
            {forecast && (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-6 h-6 text-indigo-600" />
                    <div>
                      <h2 className="text-xl font-semibold text-gray-800">48-Hour Forecast</h2>
                      <p className="text-xs text-gray-500">Predictive modeling with parameter tracking</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-500">Smog Risk Hours</p>
                    <p className="text-2xl font-bold text-indigo-600">{forecast.smog_hours}/48</p>
                  </div>
                </div>
                
                <ResponsiveContainer width="100%" height={350}>
                  <ComposedChart data={forecast.predictions?.slice(0, 48) ?? []}>
                    <XAxis 
                      dataKey="hour" 
                      stroke="#f3f4f6" 
                      tick={{ fill: '#d1d5db', fontSize: 11 }} 
                      tickFormatter={(value) => `+${value}h`}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis 
                      yAxisId="left" 
                      stroke="#f3f4f6" 
                      tick={{ fill: '#d1d5db', fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis 
                      yAxisId="right" 
                      orientation="right" 
                      stroke="#f3f4f6" 
                      tick={{ fill: '#d1d5db', fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip 
                      cursor={false}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white/98 backdrop-blur-lg p-5 rounded-2xl shadow-2xl border-2 border-indigo-200">
                              <p className="font-bold text-gray-900 mb-3 text-base">Hour +{payload[0].payload.hour}</p>
                              <div className="space-y-2">
                                {payload.map((entry, index) => (
                                  <div key={index} className="flex justify-between items-center gap-6">
                                    <span className="text-sm text-gray-600">
                                      {entry.dataKey === 'predicted_aqi' ? 'AQI' : 'PM2.5'}
                                    </span>
                                    <span className="font-bold text-base" style={{ color: entry.color }}>
                                      {entry.value.toFixed(1)}
                                      {entry.dataKey === 'predicted_pm25' ? ' µg/m³' : ''}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Area yAxisId="left" type="monotone" dataKey="predicted_aqi" stroke="#a5b4fc" fill="#ddd6fe" fillOpacity={0.5} strokeWidth={2.5} />
                    <Line yAxisId="right" type="monotone" dataKey="predicted_pm25" stroke="#fda4af" strokeWidth={2.5} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
                
                <div className="mt-4 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg">
                  <p className="text-sm text-gray-700"><strong>Summary:</strong> {forecast.summary}</p>
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'insights' && currentData && aiInsights && (
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl shadow-lg border border-purple-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-gray-800">AI-Powered Insights Engine</h2>
                  <p className="text-sm text-gray-600">Real-time trend detection & analysis</p>
                </div>
              </div>
              
              {aiInsights.aiInterpretation && (
                <div className="bg-white/70 rounded-xl p-4 mb-4">
                  <div className="flex items-start gap-2">
                    <Sparkles className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-purple-900 mb-2">🤖 AI Expert Analysis</h3>
                      <p className="text-sm text-gray-700 leading-relaxed">{aiInsights.aiInterpretation}</p>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {aiInsights.trends && (
                  <div className="bg-white/70 rounded-xl p-4">
                    <div className="flex items-start gap-2 mb-3">
                      <TrendingUp className="w-5 h-5 text-blue-600" />
                      <h3 className="font-semibold text-gray-800">Detected Trends</h3>
                    </div>
                    <div className="space-y-3">
                      <div className="bg-blue-50 rounded-lg p-3">
                        <p className="text-xs text-gray-600 mb-1">PM2.5 Trend</p>
                        <div className="flex items-center gap-2">
                          {aiInsights.trends.direction === 'up' && <ArrowUp className="w-4 h-4 text-red-600" />}
                          {aiInsights.trends.direction === 'down' && <ArrowDown className="w-4 h-4 text-green-600" />}
                          {aiInsights.trends.direction === 'stable' && <Minus className="w-4 h-4 text-gray-600" />}
                          <span className={`text-lg font-bold ${
                            aiInsights.trends.direction === 'up' ? 'text-red-600' :
                            aiInsights.trends.direction === 'down' ? 'text-green-600' : 'text-gray-600'
                          }`}>
                            {aiInsights.trends.pm25Change}%
                          </span>
                        </div>
                        <p className="text-xs text-gray-600 mt-1 capitalize">{aiInsights.trends.pm25Trend}</p>
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="bg-white/70 rounded-xl p-4">
                  <div className="flex items-start gap-2 mb-3">
                    <AlertCircle className="w-5 h-5 text-orange-600" />
                    <h3 className="font-semibold text-gray-800">Anomalies</h3>
                  </div>
                  {aiInsights.anomalies && aiInsights.anomalies.length > 0 ? (
                    <div className="space-y-2">
                      {aiInsights.anomalies.map((anomaly, idx) => (
                        <div key={idx} className={`p-2 rounded-lg ${
                          anomaly.type === 'critical' ? 'bg-red-50' :
                          anomaly.type === 'warning' ? 'bg-orange-50' : 'bg-blue-50'
                        }`}>
                          <p className="text-xs font-semibold text-gray-800">{anomaly.metric}: {anomaly.value}</p>
                          <p className="text-xs text-gray-600">{anomaly.message}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-green-600 bg-green-50 p-3 rounded-lg">
                      <CheckCircle className="w-4 h-4" />
                      <p className="text-xs">No critical anomalies</p>
                    </div>
                  )}
                </div>
              </div>
              
              {aiInsights.recommendations && aiInsights.recommendations.length > 0 && (
                <div className="bg-white/70 rounded-xl p-4 mt-4">
                  <div className="flex items-start gap-2 mb-3">
                    <CheckCircle className="w-5 h-5 text-indigo-600" />
                    <h3 className="font-semibold text-gray-800">Smart Recommendations</h3>
                  </div>
                  <div className="space-y-2">
                    {aiInsights.recommendations.map((rec, idx) => (
                      <div key={idx} className={`p-3 rounded-lg ${
                        rec.priority === 'critical' ? 'bg-red-50 border border-red-200' :
                        rec.priority === 'high' ? 'bg-orange-50 border border-orange-200' : 'bg-blue-50 border border-blue-200'
                      }`}>
                        <div className="flex items-start justify-between mb-1">
                          <p className="text-sm font-bold text-gray-800">{rec.action}</p>
                          <span className={`text-xs px-2 py-0.5 rounded-full uppercase font-bold ${
                            rec.priority === 'critical' ? 'bg-red-200 text-red-900' :
                            rec.priority === 'high' ? 'bg-orange-200 text-orange-900' : 'bg-blue-200 text-blue-900'
                          }`}>
                            {rec.priority}
                          </span>
                        </div>
                        <p className="text-xs text-gray-600">{rec.reason}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {activeTab === 'assistant' && (
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-2xl shadow-sm border border-indigo-100 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-lg flex items-center justify-center">
                  <MessageSquare className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-gray-800">AI-Powered Air Quality Assistant</h2>
                  <p className="text-sm text-gray-600">Ask questions about air pollution, smog, and health protection</p>
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 mb-4">
                <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-indigo-500" />
                  Suggested Questions
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {suggestedQuestions.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleAskQuestion(q)}
                      className="text-left px-4 py-2 bg-indigo-50 hover:bg-indigo-100 rounded-lg text-sm text-indigo-700 transition-colors"
                      disabled={isAsking}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 max-h-96 overflow-y-auto mb-4">
                {chatMessages.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <BookOpen className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                    <p className="text-sm">Ask me anything about air quality, smog formation, or protection measures!</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {chatMessages.map((msg, idx) => (
                      <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] ${msg.role === 'user' ? 'order-2' : 'order-1'}`}>
                          <div className={`rounded-lg p-3 ${
                            msg.role === 'user'
                              ? 'bg-indigo-500 text-white'
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            <p className="text-sm">{msg.content}</p>
                          </div>
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="mt-1 flex flex-wrap gap-1">
                              {msg.sources.map((source, sidx) => (
                                <span key={sidx} className="text-xs text-gray-500 bg-gray-50 px-2 py-0.5 rounded">
                                  📚 {source}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    {isAsking && (
                      <div className="flex gap-3">
                        <div className="bg-gray-100 rounded-lg p-3">
                          <div className="flex gap-1">
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              <div className="flex gap-2">
                <input
                  type="text"
                  value={userQuestion}
                  onChange={(e) => setUserQuestion(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !isAsking && handleAskQuestion(userQuestion)}
                  placeholder="Ask about air quality, health impacts, or protection measures..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-indigo-400"
                  disabled={isAsking}
                />
                <button
                  onClick={() => handleAskQuestion(userQuestion)}
                  disabled={isAsking || !userQuestion.trim()}
                  className="bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Send className="w-4 h-4" />
                  Ask
                </button>
              </div>
            </div>
            
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
              <div className="flex items-center gap-3 mb-4">
                <BookOpen className="w-6 h-6 text-indigo-600" />
                <h2 className="text-xl font-semibold text-gray-800">Knowledge Base</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {KNOWLEDGE_BASE.map((doc) => (
                  <div key={doc.id} className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-4 border border-gray-200">
                    <h3 className="font-semibold text-gray-800 mb-2">{doc.title}</h3>
                    <p className="text-sm text-gray-600 leading-relaxed">{doc.content.substring(0, 150)}...</p>
                    <button
                      onClick={() => handleAskQuestion(doc.title)}
                      className="mt-3 text-xs text-indigo-600 hover:text-indigo-700 font-medium"
                    >
                      Learn more →
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'prediction' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <Calculator className="w-6 h-6 text-indigo-600" />
                  <div>
                    <h2 className="text-xl font-semibold text-gray-800">PM2.5 & AQI Prediction</h2>
                    <p className="text-sm text-gray-500">Use ML model to predict PM2.5 levels from pollutant data</p>
                  </div>
                </div>
                <button
                  onClick={handleUseCurrentData}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                  disabled={!currentData}
                >
                  <Activity className="w-4 h-4" />
                  Use Current Data
                </button>
              </div>
              
              <form onSubmit={handlePredict} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">PM10 (μg/m³)</label>
                    <input
                      type="number"
                      step="0.01"
                      name="pm10"
                      value={predictionInputs.pm10}
                      onChange={handlePredictionInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
                      placeholder="e.g., 150.5"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">NO2 (μg/m³)</label>
                    <input
                      type="number"
                      step="0.01"
                      name="no2"
                      value={predictionInputs.no2}
                      onChange={handlePredictionInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
                      placeholder="e.g., 45.2"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">O3 (μg/m³)</label>
                    <input
                      type="number"
                      step="0.01"
                      name="o3"
                      value={predictionInputs.o3}
                      onChange={handlePredictionInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
                      placeholder="e.g., 60.8"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">CO (μg/m³)</label>
                    <input
                      type="number"
                      step="0.01"
                      name="co"
                      value={predictionInputs.co}
                      onChange={handlePredictionInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
                      placeholder="e.g., 800.0"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">SO2 (μg/m³)</label>
                    <input
                      type="number"
                      step="0.01"
                      name="so2"
                      value={predictionInputs.so2}
                      onChange={handlePredictionInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
                      placeholder="e.g., 15.3"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Temperature (°C)</label>
                    <input
                      type="number"
                      step="0.01"
                      name="temperature"
                      value={predictionInputs.temperature}
                      onChange={handlePredictionInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
                      placeholder="e.g., 25.5"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Relative Humidity (%)</label>
                    <input
                      type="number"
                      step="0.01"
                      name="relative_humidity"
                      value={predictionInputs.relative_humidity}
                      onChange={handlePredictionInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-500"
                      placeholder="e.g., 65.0"
                      required
                    />
                  </div>
                </div>
                
                <button
                  type="submit"
                  disabled={predictionLoading}
                  className="w-full bg-indigo-500 hover:bg-indigo-600 text-white py-3 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  <Calculator className="w-5 h-5" />
                  Predict PM2.5 & AQI
                  {predictionLoading && <RefreshCw className="w-4 h-4 animate-spin" />}
                </button>
              </form>
              
              {predictionError && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-start gap-2">
                  <XCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Prediction Error</p>
                    <p className="text-sm">{predictionError}</p>
                  </div>
                </div>
              )}
              
              {predictionResult && (
                <div className="mt-6 space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gradient-to-br from-rose-50 to-rose-100 rounded-xl p-6 border-2 border-rose-200">
                      <div className="flex items-center gap-2 mb-2">
                        <Cloud className="w-6 h-6 text-rose-600" />
                        <p className="text-sm text-gray-600 font-medium">Predicted PM2.5</p>
                      </div>
                      <p className="text-5xl font-bold text-rose-700 mb-1">
                        {(predictionResult.pm25 || 0).toFixed(2)}
                      </p>
                      <p className="text-sm text-gray-600">µg/m³</p>
                    </div>
                    
                    <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-xl p-6 border-2 border-indigo-200">
                      <div className="flex items-center gap-2 mb-2">
                        <Gauge className="w-6 h-6 text-indigo-600" />
                        <p className="text-sm text-gray-600 font-medium">Predicted AQI</p>
                      </div>
                      <p className="text-5xl font-bold text-indigo-700 mb-2">
                        {(predictionResult.aqi || 0).toFixed(0)}
                      </p>
                      <span className="px-3 py-1 rounded-full text-sm font-semibold text-white inline-block" style={{ backgroundColor: getAQIColor(predictionResult.aqi || 0) }}>
                        {predictionResult.health_category || 'Unknown'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-6 border border-blue-200">
                    <div className="flex items-start gap-3">
                      <Shield className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
                      <div>
                        <h3 className="font-semibold text-gray-800 mb-2">Health Advisory</h3>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {predictionResult.health_message || 'No health advisory available.'}
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                    <h3 className="font-semibold text-gray-800 mb-3 text-sm">Input Parameters Used</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(predictionResult.input_data || {}).map(([key, value]) => (
                        <div key={key} className="bg-white rounded-lg p-2 border border-gray-200">
                          <p className="text-xs text-gray-600 capitalize mb-1">{key.replace(/_/g, ' ')}</p>
                          <p className="text-sm font-semibold text-gray-800">{typeof value === 'number' ? value.toFixed(2) : value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}