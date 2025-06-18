# Import required libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import os

# Configure Gemini API with environment variable
# Logic: Use environment variable for security; updated to gemini-1.5-pro to fix 404 error.
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
try:
    model = genai.GenerativeModel('gemini-1.5-pro')
except Exception as e:
    print(f"Error initializing gemini-1.5-pro: {e}")
    model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit page configuration (unchanged)
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .big-button { font-size: 24px; padding: 20px; text-align: center; border-radius: 10px; margin: 10px; background-color: #1E88E5; color: white; }
    .disclaimer { font-size: 12px; color: #666; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# WebScraper class (unchanged)
class WebScraper:
    def __init__(self):
        # Logic: Initialize with headers to mimic a browser request and avoid bot detection.
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_news(self, query, num_articles=5):
        # Logic: Scrapes Google News for articles related to the query (e.g., "AAPL stock news").
        try:
            url = f"https://www.google.com/search?q={query}+stock+news&tbm=nws"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            for g in soup.find_all('div', class_='g')[:num_articles]:
                title = g.find('h3', class_='r')
                if title:
                    news_items.append({
                        'title': title.text,
                        'link': g.find('a')['href'] if g.find('a') else ''
                    })
            return news_items
        except Exception as e:
            return [{'title': f'Error fetching news: {str(e)}', 'link': ''}]

# FinancialChatbot class (unchanged)
class FinancialChatbot:
    def __init__(self, model):
        self.model = model
        self.context = """You are a knowledgeable financial advisor..."""

    def get_response(self, user_input):
        try:
            prompt = f"{self.context}\nUser: {user_input}\nAdvisor:"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

# StockAnalyzer class (updated news display)
class StockAnalyzer:
    def __init__(self):
        self.scraper = WebScraper()
        self.model = model

    def get_stock_data(self, symbol, period='1y'):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            return hist, stock.info
        except Exception as e:
            return None, None

    def create_technical_analysis(self, data):
        if data is None or len(data) < 50:
            return None
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        return data

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def plot_stock_data(self, data, symbol):
        if data is None:
            return None
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='OHLC'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA50', line=dict(color='blue')))
        fig.update_layout(title=f'{symbol} Stock Price Analysis', yaxis_title='Price', xaxis_title='Date', template='plotly_dark')
        return fig

# Main function (updated news display and key fixes)
def main():
    st.title("AI Financial Assistant 💰")
    try:
        chatbot = FinancialChatbot(model)
        analyzer = StockAnalyzer()
        st.markdown("<div style='text-align: center;'><h2>Choose Your Financial Journey</h2></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Chat with Financial Advisor", key="chat_button", help="Get personalized financial advice"):
                st.session_state.mode = "chat"
        with col2:
            if st.button("📊 Stock Analysis", key="analysis_button", help="Analyze specific
