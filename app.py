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
genai.configure(api_key='AIzaSyD_o1EvxljYD3_rKVIFhL4KqxNyV9z7yWI')
# Updated model name - try these in order of preference
try:
    model = genai.GenerativeModel('gemini-1.5-flash')  # Most current model
except:
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')  # Alternative current model
    except:
        model = genai.GenerativeModel('gemini-1.0-pro')  # Fallback model

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
        margin-top: 20px;
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 4px solid #17a2b8;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
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
            # Using a more reliable news source
            url = f"https://finance.yahoo.com/quote/{query}/news"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Try to find news items (this is a simplified approach)
            for item in soup.find_all('h3')[:num_articles]:
                if item.text.strip():
                    news_items.append({
                        'title': item.text.strip(),
                        'link': ''
                    })
            
            # If no news found, return placeholder
            if not news_items:
                news_items = [
                    {'title': f'Recent {query} market activity', 'link': ''},
                    {'title': f'{query} trading volume analysis', 'link': ''},
                    {'title': f'{query} technical indicators update', 'link': ''}
                ]
            
            return news_items
        except Exception as e:
            return [
                {'title': f'News service temporarily unavailable', 'link': ''},
                {'title': f'Please check financial news websites for {query} updates', 'link': ''}
            ]

class FinancialChatbot:
    def __init__(self, model):
        self.model = model
        self.context = """You are a knowledgeable financial advisor. Provide clear, step-by-step guidance 
        on investing and financial planning. Always include disclaimers about financial risks. Focus on 
        educational content and avoid making specific investment recommendations. Keep responses concise 
        and practical."""

    def get_response(self, user_input):
        try:
            prompt = f"{self.context}\n\nUser Question: {user_input}\n\nFinancial Advisor Response:"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties. Here's some general advice: Always diversify your investments, do thorough research before investing, and consider consulting with a qualified financial advisor for personalized guidance. Error details: {str(e)}"

class StockAnalyzer:
    def __init__(self):
        self.scraper = WebScraper()
        self.model = model

    def get_stock_data(self, symbol, period='1y'):
        try:
            stock = yf.Ticker(symbol.upper())
            hist = stock.history(period=period)
            info = stock.info
            
            if hist.empty:
                return None, None
                
            return hist, info
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None

    def create_technical_analysis(self, data):
        if data is None or len(data) < 50:
            return data

        try:
            # Calculate technical indicators
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            return data
        except Exception as e:
            st.warning(f"Error calculating technical indicators: {str(e)}")
            return data

    @staticmethod
    def calculate_rsi(prices, period=14):
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

    def plot_stock_data(self, data, symbol):
        if data is None or data.empty:
            return None

        try:
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ))
            
            # Add SMAs if they exist
            if 'SMA20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    name='SMA20',
                    line=dict(color='orange', width=2)
                ))
            
            if 'SMA50' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA50'],
                    name='SMA50',
                    line=dict(color='blue', width=2)
                ))
            
            fig.update_layout(
                title=f'{symbol.upper()} Stock Price Analysis',
                yaxis_title='Price ($)',
                xaxis_title='Date',
                template='plotly_white',
                height=600,
                showlegend=True
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None

def test_gemini_model():
    """Test which Gemini model is available"""
    models_to_try = [
        'gemini-1.5-flash',
        'gemini-1.5-pro', 
        'gemini-1.0-pro',
        'gemini-pro-latest'
    ]
    
    for model_name in models_to_try:
        try:
            test_model = genai.GenerativeModel(model_name)
            test_response = test_model.generate_content("Hello")
            st.success(f"✅ Successfully connected to {model_name}")
            return test_model
        except Exception as e:
            st.warning(f"❌ {model_name} not available: {str(e)}")
            continue
    
    st.error("❌ No Gemini models are available. Please check your API key and internet connection.")
    return None

def main():
    st.title("🤖 AI Financial Assistant 💰")
    
    # Test Gemini model connection
    st.sidebar.header("🔧 System Status")
    if st.sidebar.button("Test Gemini Connection"):
        with st.sidebar:
            working_model = test_gemini_model()
            if working_model:
                global model
                model = working_model
    
    try:
        # Initialize components
        chatbot = FinancialChatbot(model)
        analyzer = StockAnalyzer()
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        mode = st.sidebar.radio(
            "Choose your tool:",
            ["Home", "💬 Financial Advisor Chat", "📊 Stock Analysis"]
        )
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Home Page
        if mode == "Home":
            st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <h2>Welcome to Your AI Financial Assistant</h2>
                    <p>Get personalized financial advice and comprehensive stock analysis powered by AI</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    ### 💬 Financial Advisor Chat
                    - Get personalized investment advice
                    - Learn about financial planning
                    - Ask questions about market trends
                    - Receive educational content
                """)
            
            with col2:
                st.markdown("""
                    ### 📊 Stock Analysis
                    - Real-time stock data analysis
                    - Technical indicators (RSI, SMA)
                    - Interactive price charts
                    - Latest market news
                """)
        
        # Chat Mode
        elif mode == "💬 Financial Advisor Chat":
            st.header("💬 Financial Advisor Chat")
            
            # Chat input
            user_input = st.text_input(
                "Ask me anything about investing and finance:",
                placeholder="e.g., How should I start investing with $1000?",
                key="user_input"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                send_button = st.button("Send 📨", key="send_button")
            with col2:
                if st.button("Clear Chat History 🗑️"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if send_button and user_input:
                with st.spinner("Getting financial advice..."):
                    response = chatbot.get_response(user_input)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Advisor", response))
                st.rerun()
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "You":
                        st.markdown(f"**🙋 You:** {message}")
                    else:
                        st.markdown(f"**🤖 Advisor:** {message}")
                    st.markdown("---")
        
        # Analysis Mode
        elif mode == "📊 Stock Analysis":
            st.header("📊 Stock Analysis Dashboard")
            
            # Stock symbol input
            col1, col2 = st.columns([3, 1])
            with col1:
                symbol = st.text_input(
                    "Enter Stock Symbol:",
                    placeholder="e.g., AAPL, GOOGL, MSFT",
                    key="symbol_input"
                ).upper()
            with col2:
                period = st.selectbox(
                    "Time Period:",
                    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                    index=3
                )
            
            if symbol:
                with st.spinner(f"Analyzing {symbol}..."):
                    # Get stock data
                    data, info = analyzer.get_stock_data(symbol, period)
                    
                    if data is not None and not data.empty:
                        # Technical analysis
                        data = analyzer.create_technical_analysis(data)
                        
                        # Display basic info
                        if info:
                            st.subheader(f"{info.get('longName', symbol)} ({symbol})")
                            st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
                        
                        # Plot stock data
                        fig = analyzer.plot_stock_data(data, symbol)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display key metrics
                        st.subheader("Key Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        current_price = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        daily_change = current_price - prev_close
                        daily_change_pct = (daily_change / prev_close) * 100
                        
                        with col1:
                            st.metric(
                                "Current Price",
                                f"${current_price:.2f}",
                                f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)"
                            )
                        with col2:
                            if 'RSI' in data.columns and not data['RSI'].isna().all():
                                rsi_value = data['RSI'].iloc[-1]
                                st.metric("RSI (14)", f"{rsi_value:.1f}")
                            else:
                                st.metric("RSI (14)", "N/A")
                        with col3:
                            volume = data['Volume'].iloc[-1]
                            st.metric("Volume", f"{volume:,.0f}")
                        with col4:
                            high_52w = data['High'].max()
                            low_52w = data['Low'].min()
                            st.metric("52W High", f"${high_52w:.2f}")
                        
                        # News section
                        st.subheader("📰 Recent News & Updates")
                        news_items = analyzer.scraper.get_news(symbol)
                        
                        for i, item in enumerate(news_items[:5], 1):
                            st.markdown(f"**{i}.** {item['title']}")
                        
                        # AI Analysis
                        st.subheader("🤖 AI Technical Analysis")
                        try:
                            rsi_text = f"{data['RSI'].iloc[-1]:.1f}" if 'RSI' in data.columns and not data['RSI'].isna().all() else "N/A"
                            sma20_diff = data['Close'].iloc[-1] - data['SMA20'].iloc[-1] if 'SMA20' in data.columns else 0
                            sma50_diff = data['Close'].iloc[-1] - data['SMA50'].iloc[-1] if 'SMA50' in data.columns else 0
                            
                            analysis_prompt = f"""Provide a brief technical analysis for {symbol} stock based on:
                            - Current Price: ${current_price:.2f}
                            - Daily Change: {daily_change_pct:+.2f}%
                            - RSI: {rsi_text}
                            - Price vs 20-day SMA: {sma20_diff:+.2f}
                            - Price vs 50-day SMA: {sma50_diff:+.2f}
                            
                            Keep it concise and educational. Include risk disclaimers."""
                            
                            analysis_text = analyzer.model.generate_content(analysis_prompt).text
                            st.write(analysis_text)
                        except Exception as e:
                            st.write("Technical analysis temporarily unavailable. Please refer to the charts and metrics above for your analysis.")
                    
                    else:
                        st.error(f"❌ Could not fetch data for '{symbol}'. Please check the symbol and try again.")
        
        # Disclaimer
        st.markdown("""
            <div class='disclaimer'>
                <strong>⚠️ Important Disclaimer:</strong> This AI Financial Assistant is for educational and informational purposes only. 
                It does not constitute financial advice, investment recommendations, or professional guidance. 
                Stock market investments carry inherent risks, and past performance does not guarantee future results. 
                Always conduct thorough research and consult with qualified financial advisors before making investment decisions. 
                The creators of this tool are not responsible for any financial losses or decisions made based on this information.
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, check your internet connection and API keys.")

if __name__ == "__main__":
    main()
