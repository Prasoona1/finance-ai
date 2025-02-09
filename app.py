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

# Initialize session state for portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

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

# Enhanced Custom CSS with portfolio styling
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
    .portfolio-table {
        margin-top: 1rem;
        border-collapse: collapse;
        width: 100%;
    }
    .portfolio-table th, .portfolio-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #333;
    }
    .portfolio-table th {
        background-color: #1b5e20;
        color: white;
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
    @staticmethod
    def add_position(symbol, shares, purchase_price, purchase_date):
        try:
            # Validate the stock symbol
            stock = yf.Ticker(symbol)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            position = {
                'symbol': symbol.upper(),
                'shares': float(shares),
                'purchase_price': float(purchase_price),
                'purchase_date': purchase_date,
                'current_price': current_price,
                'market_value': current_price * float(shares),
                'gain_loss': (current_price - float(purchase_price)) * float(shares),
                'gain_loss_percent': ((current_price - float(purchase_price)) / float(purchase_price)) * 100
            }
            
            st.session_state.portfolio.append(position)
            return True, "Position added successfully!"
        except Exception as e:
            return False, f"Error adding position: {str(e)}"

    @staticmethod
    def remove_position(index):
        try:
            st.session_state.portfolio.pop(index)
            return True, "Position removed successfully!"
        except Exception as e:
            return False, f"Error removing position: {str(e)}"

    @staticmethod
    def update_portfolio_prices():
        updated_portfolio = []
        for position in st.session_state.portfolio:
            try:
                stock = yf.Ticker(position['symbol'])
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                position['current_price'] = current_price
                position['market_value'] = current_price * position['shares']
                position['gain_loss'] = (current_price - position['purchase_price']) * position['shares']
                position['gain_loss_percent'] = ((current_price - position['purchase_price']) / position['purchase_price']) * 100
                updated_portfolio.append(position)
            except Exception as e:
                st.error(f"Error updating {position['symbol']}: {str(e)}")
                updated_portfolio.append(position)
        
        st.session_state.portfolio = updated_portfolio

    @staticmethod
    def get_portfolio_summary():
        total_value = sum(position['market_value'] for position in st.session_state.portfolio)
        total_cost = sum(position['purchase_price'] * position['shares'] for position in st.session_state.portfolio)
        total_gain_loss = sum(position['gain_loss'] for position in st.session_state.portfolio)
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_percent': (total_gain_loss / total_cost * 100) if total_cost > 0 else 0,
            'position_count': len(st.session_state.portfolio)
        }

    @staticmethod
    def get_portfolio_allocation():
        if not st.session_state.portfolio:
            return pd.DataFrame()
        
        total_value = sum(position['market_value'] for position in st.session_state.portfolio)
        allocations = [
            {
                'symbol': position['symbol'],
                'allocation': (position['market_value'] / total_value * 100) if total_value > 0 else 0
            }
            for position in st.session_state.portfolio
        ]
        return pd.DataFrame(allocations)

def create_portfolio_pie_chart(allocations):
    if allocations.empty:
        return None
    
    fig = px.pie(
        allocations,
        values='allocation',
        names='symbol',
        title='Portfolio Allocation',
        hole=0.3
    )
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_x=0.5
    )
    return fig

def main():
    st.title("💎 Wealth Dashboard")
    
    # Initialize components
    analyzer = StockAnalyzer()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Portfolio Management", "Stock Analysis", "Market Overview"])
    
    if page == "Portfolio Management":
        st.header("Portfolio Management")
        
        # Add Position Form
        with st.expander("Add New Position"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                symbol = st.text_input("Stock Symbol").upper()
            with col2:
                shares = st.number_input("Number of Shares", min_value=0.0, step=0.01)
            with col3:
                purchase_price = st.number_input("Purchase Price ($)", min_value=0.0, step=0.01)
            with col4:
                purchase_date = st.date_input("Purchase Date")
            
            if st.button("Add Position"):
                success, message = Portfolio.add_position(symbol, shares, purchase_price, purchase_date)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Portfolio Summary
        if st.session_state.portfolio:
            Portfolio.update_portfolio_prices()
            summary = Portfolio.get_portfolio_summary()
            
            # Summary Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Portfolio Value", f"${summary['total_value']:,.2f}")
            with col2:
                st.metric("Total Gain/Loss", 
                         f"${summary['total_gain_loss']:,.2f}",
                         f"{summary['total_gain_loss_percent']:,.2f}%")
            with col3:
                st.metric("Number of Positions", summary['position_count'])
            
            # Portfolio Allocation Chart
            allocations = Portfolio.get_portfolio_allocation()
            if not allocations.empty:
                fig = create_portfolio_pie_chart(allocations)
                st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio Positions Table
            st.subheader("Portfolio Positions")
            for idx, position in enumerate(st.session_state.portfolio):
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.write(f"**{position['symbol']}**")
                    col2.write(f"{position['shares']} shares")
                    col3.write(f"${position['current_price']:.2f}")
                    col4.write(f"${position['market_value']:.2f}")
                    col5.write(f"{position['gain_loss_percent']:.2f}%")
                    if col6.button("Remove", key=f"remove_{idx}"):
                        success, message = Portfolio.remove_position(idx)
                        if success:
                            st.success(message)
                            st.experimental_rerun()
                        else:
                            st.error(message)
                    st.divider()
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
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                            st.write(f"**Market Cap:** ${info.get('marketCap', 0):,.2f}")
                        with col2:
                            st.write(f"**52 Week High:** ${info.get('fiftyTwoWeekHigh', 0):,.2f}")
                            st.write(f"**52 Week Low:** ${info.get('fiftyTwoWeekLow', 0):,.2f}")
                            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")

    elif page == "Market Overview":
        st.header("Market Overview")
        
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
        
        # Market News
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
