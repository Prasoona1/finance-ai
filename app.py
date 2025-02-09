import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import os

# Configure Gemini API
genai.configure(api_key='YOUR_GEMINI_API_KEY')
model = genai.GenerativeModel('gemini-pro')

# Set page config
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #1E88E5;
        --accent-color: #FFD700;
        --background-dark: #0E1117;
        --card-background: #1E1E1E;
    }

    .main {
        background-color: var(--background-dark);
        color: white;
    }

    .stButton>button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }

    .metric-card {
        background: linear-gradient(145deg, var(--card-background), #2C2C2C);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stats-card {
        background: rgba(30, 136, 229, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--secondary-color);
        margin: 0.5rem 0;
    }

    .news-card {
        background: rgba(46, 125, 50, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
    }

    /* Chat styling */
    .user-message {
        background: rgba(30, 136, 229, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--secondary-color);
    }

    .advisor-message {
        background: rgba(46, 125, 50, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
    }

    .big-number {
        font-size: 28px;
        font-weight: bold;
        color: var(--accent-color);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .disclaimer {
        font-size: 12px;
        color: #888;
        font-style: italic;
        padding: 15px;
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.2);
        border-left: 3px solid var(--accent-color);
    }

    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(45deg, #81c784, #64b5f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }

    /* Metrics animation */
    .stMetric {
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Add the existing classes (WebScraper, FinancialChatbot, StockAnalyzer) here...
[Previous classes remain the same]

def create_market_summary():
    indices = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT'
    }
    
    index_data = []
    for name, symbol in indices.items():
        try:
            data = yf.download(symbol, period='2d')
            if not data.empty:
                current = data['Close'][-1]
                prev = data['Close'][-2]
                change = ((current - prev) / prev) * 100
                index_data.append({
                    'name': name,
                    'price': current,
                    'change': change
                })
        except Exception:
            continue
    
    return index_data

def main():
    st.title("🌟 AI Financial Assistant")
    
    try:
        # Initialize components
        chatbot = FinancialChatbot(model)
        analyzer = StockAnalyzer()
        
        # Market Summary Section
        st.markdown("### 📈 Market Overview")
        market_data = create_market_summary()
        cols = st.columns(len(market_data))
        for idx, data in enumerate(market_data):
            with cols[idx]:
                delta_color = "normal" if data['change'] == 0 else ("inverse" if data['change'] < 0 else "normal")
                st.metric(
                    data['name'],
                    f"${data['price']:,.2f}",
                    f"{data['change']:+.2f}%",
                    delta_color=delta_color
                )
        
        # Main menu with enhanced styling
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h2>✨ Choose Your Financial Journey</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💬 Chat with AI Advisor", key="chat_button", help="Get personalized financial advice"):
                st.session_state.mode = "chat"
        
        with col2:
            if st.button("📊 Advanced Stock Analysis", key="analysis_button", help="Analyze specific stocks"):
                st.session_state.mode = "analysis"
        
        # Initialize session state
        if 'mode' not in st.session_state:
            st.session_state.mode = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Enhanced Chat Mode
        if st.session_state.mode == "chat":
            st.markdown("### 💬 Financial Advisor Chat")
            
            user_input = st.text_input("💭 Ask me anything about investing:", key="user_input")
            
            if st.button("Send Message", key="send_button"):
                if user_input:
                    response = chatbot.get_response(user_input)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Advisor", response))
            
            # Enhanced chat history display
            for role, message in st.session_state.chat_history:
                if role == "You":
                    st.markdown(f"""
                        <div class="user-message">
                            <strong>You:</strong><br>{message}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="advisor-message">
                            <strong>AI Advisor:</strong><br>{message}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Enhanced Analysis Mode
        elif st.session_state.mode == "analysis":
            st.markdown("### 📊 Stock Analysis Dashboard")
            
            # Stock input with period selection
            col1, col2 = st.columns([3, 1])
            with col1:
                symbol = st.text_input("🔍 Enter Stock Symbol (e.g., AAPL):", key="symbol_input")
            with col2:
                period = st.selectbox("📅 Time Period", 
                                    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                    index=3)
            
            if symbol:
                # Get stock data
                data, info = analyzer.get_stock_data(symbol, period=period)
                
                if data is not None:
                    # Technical analysis
                    data = analyzer.create_technical_analysis(data)
                    
                    # Enhanced stock information
                    if info:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("""
                                <div class="stats-card">
                                    <h4>Company Info</h4>
                                    <p>Sector: {}<br>Industry: {}</p>
                                </div>
                            """.format(info.get('sector', 'N/A'), info.get('industry', 'N/A')), 
                            unsafe_allow_html=True)
                        with col2:
                            st.markdown("""
                                <div class="stats-card">
                                    <h4>Market Stats</h4>
                                    <p>Market Cap: ${:,.0f}M<br>P/E Ratio: {}</p>
                                </div>
                            """.format(
                                info.get('marketCap', 0) / 1_000_000,
                                info.get('trailingPE', 'N/A')
                            ), unsafe_allow_html=True)
                        with col3:
                            st.markdown("""
                                <div class="stats-card">
                                    <h4>52 Week Range</h4>
                                    <p>High: ${:,.2f}<br>Low: ${:,.2f}</p>
                                </div>
                            """.format(
                                info.get('fiftyTwoWeekHigh', 0),
                                info.get('fiftyTwoWeekLow', 0)
                            ), unsafe_allow_html=True)
                    
                    # Plot stock data
                    fig = analyzer.plot_stock_data(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", 
                                f"${data['Close'][-1]:.2f}",
                                f"{((data['Close'][-1] - data['Open'][-1]) / data['Open'][-1] * 100):.2f}%")
                    with col2:
                        st.metric("RSI", 
                                f"{data['RSI'][-1]:.2f}",
                                "Oversold" if data['RSI'][-1] < 30 else ("Overbought" if data['RSI'][-1] > 70 else "Neutral"))
                    with col3:
                        sma_diff = ((data['Close'][-1] - data['SMA20'][-1]) / data['SMA20'][-1] * 100)
                        st.metric("vs SMA20", 
                                f"{sma_diff:+.2f}%",
                                "Above" if sma_diff > 0 else "Below")
                    with col4:
                        volume_change = ((data['Volume'][-1] - data['Volume'][-2]) / data['Volume'][-2] * 100)
                        st.metric("Volume Change", 
                                f"{data['Volume'][-1]:,.0f}",
                                f"{volume_change:+.2f}%")
                    
                    # News section with enhanced styling
                    st.markdown("### 📰 Recent News")
                    news_items = analyzer.scraper.get_news(f"{symbol} stock")
                    for item in news_items:
                        st.markdown(f"""
                            <div class="news-card">
                                📱 {item['title']}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Enhanced analysis summary
                    st.markdown("### 🔍 Technical Analysis")
                    analysis_text = model.generate_content(
                        f"Provide a brief technical analysis summary for {symbol} stock based on: "
                        f"Current RSI: {data['RSI'][-1]:.2f}, "
                        f"Price vs SMA20: {data['Close'][-1] - data['SMA20'][-1]:.2f}, "
                        f"Price vs SMA50: {data['Close'][-1] - data['SMA50'][-1]:.2f}"
                    ).text
                    st.markdown(f"""
                        <div class="stats-card">
                            {analysis_text}
                        </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error("❌ Error fetching stock data. Please check the symbol and try again.")
        
        # Enhanced footer
        st.markdown("""
            <div class="disclaimer">
                ⚠️ Disclaimer: This is an AI-powered financial assistant for educational purposes only. 
                Historical performance does not guarantee future results. Always conduct thorough research 
                and consult with qualified financial advisors before making investment decisions.
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
