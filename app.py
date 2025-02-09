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

# Configure Gemini API
genai.configure(api_key='AIzaSyDfgOLS8K9F7pQgUZtz_6rNa-qu_LKzLls')
model = genai.GenerativeModel('gemini-pro')

# Set page config
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-button {
        font-size: 24px;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin: 10px;
        background-color: #1E88E5;
        color: white;
    }
    .disclaimer {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_news(self, query, num_articles=5):
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

class FinancialChatbot:
    def __init__(self, model):  # Fixed from _init_ to __init__
        self.model = model
        self.context = """You are a knowledgeable financial advisor. Provide clear, step-by-step guidance 
        on investing and financial planning. Always include disclaimers about financial risks. Focus on 
        educational content and avoid making specific investment recommendations."""

    def get_response(self, user_input):
        try:
            prompt = f"{self.context}\nUser: {user_input}\nAdvisor:"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"


      
        
        return data

 class StockAnalyzer:
    def __init__(self):
        self.scraper = WebScraper()
        self.model = model

    def get_stock_data(self, symbol, period='1y'):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            if hist.empty:
                return None, None
            return hist, stock.info
        except Exception as e:
            return None, None

    def create_technical_analysis(self, data):
        if data is None or len(data) < 50:
            return None

        # Calculate technical indicators
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
        if data is None or data.empty:
            return None

        # Create figure with secondary y-axis
        fig = go.Figure()

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            )
        )

        # Add Moving averages
        if 'SMA20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    name='20-day SMA',
                    line=dict(color='orange', width=1)
                )
            )

        if 'SMA50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA50'],
                    name='50-day SMA',
                    line=dict(color='blue', width=1)
                )
            )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price Analysis',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,  # Disable rangeslider
            height=600,  # Set height
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig
def main():
    st.title("AI Financial Assistant 💰")
    
    try:
        # Initialize components
        chatbot = FinancialChatbot(model)
        analyzer = StockAnalyzer()
        
        # Main menu
        st.markdown("""
            <div style='text-align: center;'>
                <h2>Choose Your Financial Journey</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💬 Chat with Financial Advisor", key="chat_button", help="Get personalized financial advice"):
                st.session_state.mode = "chat"
        
        with col2:
            if st.button("📊 Stock Analysis", key="analysis_button", help="Analyze specific stocks"):
                st.session_state.mode = "analysis"
        
        # Initialize session state
        if 'mode' not in st.session_state:
            st.session_state.mode = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat Mode
        if st.session_state.mode == "chat":
            st.header("Financial Advisor Chat")
            
            user_input = st.text_input("Ask me anything about investing:", key="user_input")
            
            if st.button("Send", key="send_button"):
                if user_input:
                    response = chatbot.get_response(user_input)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Advisor", response))
            
            # Display chat history
            for role, message in st.session_state.chat_history:
                if role == "You":
                    st.markdown(f"*You:* {message}")
                else:
                    st.markdown(f"*Advisor:* {message}")
            
            st.markdown("""
                <div class='disclaimer'>
                    Disclaimer: This is an AI-powered financial assistant for educational purposes only. 
                    Always consult with a qualified financial advisor before making investment decisions.
                </div>
            """, unsafe_allow_html=True)
        
        # Analysis Mode
        elif st.session_state.mode == "analysis":
            st.header("Stock Analysis Dashboard")
            
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", key="symbol_input")
            
            if symbol:
                # Get stock data
                data, info = analyzer.get_stock_data(symbol)
                
                if data is not None:
                    # Technical analysis
                    data = analyzer.create_technical_analysis(data)
                    
                    # Plot stock data
                    fig = analyzer.plot_stock_data(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display key statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${data['Close'][-1]:.2f}")
                    with col2:
                        st.metric("RSI", f"{data['RSI'][-1]:.2f}")
                    with col3:
                        daily_return = ((data['Close'][-1] - data['Close'][-2]) / data['Close'][-2]) * 100
                        st.metric("Daily Return", f"{daily_return:.2f}%")
                    
                    # News section
                    st.subheader("Recent News")
                    news_items = analyzer.scraper.get_news(f"{symbol} stock")
                    for item in news_items:
                        st.markdown(f"* {item['title']}")
                    
                    # Analysis summary
                    st.subheader("Technical Analysis Summary")
                    analysis_text = model.generate_content(
                        f"Provide a brief technical analysis summary for {symbol} stock based on: "
                        f"Current RSI: {data['RSI'][-1]:.2f}, "
                        f"Price vs SMA20: {data['Close'][-1] - data['SMA20'][-1]:.2f}, "
                        f"Price vs SMA50: {data['Close'][-1] - data['SMA50'][-1]:.2f}"
                    ).text
                    st.write(analysis_text)
                
                else:
                    st.error("Error fetching stock data. Please check the symbol and try again.")
            
            st.markdown("""
                <div class='disclaimer'>
                    Disclaimer: This analysis is for educational purposes only. Past performance does not 
                    guarantee future results. Always do your own research and consult with a qualified 
                    financial advisor before making investment decisions.
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":  # Fixed from _main_ to __main__
    main()
