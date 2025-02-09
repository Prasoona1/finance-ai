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
genai.configure(api_key='YOUR_GEMINI_API_KEY')
model = genai.GenerativeModel('gemini-pro')

# Set page config
st.set_page_config(
    page_title="Wealth Dashboard 💰",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    .portfolio-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #2e7d32;
        margin: 1rem 0;
    }
    .big-number {
        font-size: 24px;
        font-weight: bold;
        color: #4caf50;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #81c784;
    }
    .disclaimer {
        font-size: 12px;
        color: #666;
        font-style: italic;
        padding: 10px;
        border-left: 3px solid #2e7d32;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

class Portfolio:
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        
    def add_position(self, symbol, shares, purchase_price):
        position = {
            'symbol': symbol,
            'shares': shares,
            'purchase_price': purchase_price,
            'date_added': datetime.now().strftime('%Y-%m-%d')
        }
        st.session_state.portfolio.append(position)
    
    def remove_position(self, index):
        st.session_state.portfolio.pop(index)
    
    def get_portfolio_value(self):
        total_value = 0
        for position in st.session_state.portfolio:
            stock = yf.Ticker(position['symbol'])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            value = current_price * position['shares']
            total_value += value
        return total_value
    
    def get_portfolio_performance(self):
        performance_data = []
        total_gain_loss = 0
        
        for position in st.session_state.portfolio:
            stock = yf.Ticker(position['symbol'])
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            cost_basis = position['purchase_price'] * position['shares']
            current_value = current_price * position['shares']
            gain_loss = current_value - cost_basis
            gain_loss_percent = (gain_loss / cost_basis) * 100 if cost_basis != 0 else 0
            
            performance_data.append({
                'symbol': position['symbol'],
                'shares': position['shares'],
                'current_price': current_price,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_percent': gain_loss_percent
            })
            
            total_gain_loss += gain_loss
            
        return performance_data, total_gain_loss

class StockAnalyzer:
    def __init__(self):
        self.scraper = WebScraper()

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
        data['MACD'] = self.calculate_macd(data['Close'])
        data['Signal'] = self.calculate_macd_signal(data['MACD'])
        
        return data

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2

    @staticmethod
    def calculate_macd_signal(macd, signal_period=9):
        return macd.ewm(span=signal_period, adjust=False).mean()

    def plot_stock_data(self, data, symbol):
        if data is None:
            return None

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
        
        # Add technical indicators
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA20'],
            name='SMA20',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA50'],
            name='SMA50',
            line=dict(color='blue', width=1)
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark',
            height=600,
            margin=dict(l=50, r=50, t=100, b=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

def main():
    st.title("💎 Wealth Dashboard")
    
    # Initialize components
    portfolio = Portfolio()
    analyzer = StockAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Portfolio Dashboard", "Stock Analysis", "Market News"])
    
    if page == "Portfolio Dashboard":
        st.header("Portfolio Management")
        
        # Add new position form
        with st.expander("Add New Position"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_symbol = st.text_input("Stock Symbol")
            with col2:
                shares = st.number_input("Number of Shares", min_value=0.0)
            with col3:
                purchase_price = st.number_input("Purchase Price", min_value=0.0)
            
            if st.button("Add Position"):
                portfolio.add_position(new_symbol, shares, purchase_price)
                st.success(f"Added {shares} shares of {new_symbol}")
        
        # Portfolio Summary
        st.subheader("Portfolio Summary")
        total_value = portfolio.get_portfolio_value()
        performance_data, total_gain_loss = portfolio.get_portfolio_performance()
        
        # Portfolio Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}", 
                     delta=f"{(total_gain_loss/total_value)*100:.1f}%" if total_value > 0 else "0%")
        with col3:
            st.metric("Number of Positions", len(st.session_state.portfolio))
        
        # Portfolio Positions Table
        st.subheader("Portfolio Positions")
        if performance_data:
            df = pd.DataFrame(performance_data)
            st.dataframe(df.style.format({
                'current_price': '${:.2f}',
                'cost_basis': '${:.2f}',
                'current_value': '${:.2f}',
                'gain_loss': '${:.2f}',
                'gain_loss_percent': '{:.1f}%'
            }))
        else:
            st.info("No positions in portfolio. Add some positions to get started!")

    elif page == "Stock Analysis":
        st.header("Stock Analysis")
        
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
        
        if symbol:
            data, info = analyzer.get_stock_data(symbol)
            
            if data is not None:
                # Technical analysis
                data = analyzer.create_technical_analysis(data)
                
                # Stock chart
                fig = analyzer.plot_stock_data(data, symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Indicators
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${data['Close'][-1]:.2f}")
                with col2:
                    st.metric("RSI", f"{data['RSI'][-1]:.2f}")
                with col3:
                    st.metric("MACD", f"{data['MACD'][-1]:.2f}")
                with col4:
                    daily_return = ((data['Close'][-1] - data['Close'][-2]) / data['Close'][-2]) * 100
                    st.metric("Daily Return", f"{daily_return:.2f}%")
                
                # Company Info
                if info:
                    with st.expander("Company Information"):
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.write(f"**Market Cap:** ${info.get('marketCap', 0):,.2f}")
                        st.write(f"**52 Week High:** ${info.get('fiftyTwoWeekHigh', 0):,.2f}")
                        st.write(f"**52 Week Low:** ${info.get('fiftyTwoWeekLow', 0):,.2f}")

    elif page == "Market News":
        st.header("Market News")
        
        # Market indices
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC'
        }
        
        col1, col2, col3 = st.columns(3)
        for idx, (name, symbol) in enumerate(indices.items()):
            data = yf.download(symbol, period='1d')
            if not data.empty:
                with [col1, col2, col3][idx]:
                    current_price = data['Close'][-1]
                    prev_price = data['Open'][0]
                    change = ((current_price - prev_price) / prev_price) * 100
                    st.metric(name, f"{current_price:,.2f}", f"{change:+.2f}%")
        
        # News feed
        st.subheader("Latest Market News")
        scraper = WebScraper()
        news_items = scraper.get_news("stock market", num_articles=10)
        for item in news_items:
            st.markdown(f"* {item['title']}")

    # Footer
    st.markdown("""
        <div class='disclaimer'>
            Disclaimer: This dashboard is for educational purposes only. All investment decisions carry risk. 
            Past performance does not guarantee future results. Always consult with a qualified financial advisor 
            before making investment decisions.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
