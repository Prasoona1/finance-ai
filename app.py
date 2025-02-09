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
genai.configure(api_key='AIzaSyBWeq2u8f-kwEVhmTFnDsGr9jza3tSmB1s')
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
    .stApp {
        background-color: #f5f7fa;
    }
    .big-button {
        font-size: 20px;
        padding: 15px;
        text-align: center;
        border-radius: 10px;
        margin: 10px;
        background-color: #1E88E5;
        color: white;
        transition: all 0.3s;
    }
    .big-button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
    }
    .dashboard-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .disclaimer {
        font-size: 12px;
        color: #666;
        font-style: italic;
        padding: 10px;
        border-top: 1px solid #eee;
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Add Portfolio Management Class
class PortfolioManager:
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}

    def add_position(self, symbol, shares, purchase_price):
        if symbol not in st.session_state.portfolio:
            st.session_state.portfolio[symbol] = {
                'shares': shares,
                'purchase_price': purchase_price,
                'date_added': datetime.now().strftime('%Y-%m-%d')
            }
        else:
            # Average down/up the purchase price
            total_shares = st.session_state.portfolio[symbol]['shares'] + shares
            total_cost = (st.session_state.portfolio[symbol]['shares'] * st.session_state.portfolio[symbol]['purchase_price'] +
                         shares * purchase_price)
            st.session_state.portfolio[symbol]['shares'] = total_shares
            st.session_state.portfolio[symbol]['purchase_price'] = total_cost / total_shares

    def remove_position(self, symbol):
        if symbol in st.session_state.portfolio:
            del st.session_state.portfolio[symbol]

    def get_portfolio_value(self):
        total_value = 0
        positions = []
        
        for symbol, data in st.session_state.portfolio.items():
            try:
                current_price = yf.Ticker(symbol).history(period='1d')['Close'][-1]
                position_value = data['shares'] * current_price
                cost_basis = data['shares'] * data['purchase_price']
                gain_loss = position_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis != 0 else 0
                
                positions.append({
                    'symbol': symbol,
                    'shares': data['shares'],
                    'purchase_price': data['purchase_price'],
                    'current_price': current_price,
                    'position_value': position_value,
                    'gain_loss': gain_loss,
                    'gain_loss_pct': gain_loss_pct
                })
                
                total_value += position_value
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return total_value, positions

[Previous classes remain the same...]

def main():
    st.title("AI Financial Assistant 💰")
    
    # Initialize components
    chatbot = FinancialChatbot(model)
    analyzer = StockAnalyzer()
    portfolio_manager = PortfolioManager()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Dashboard", "Portfolio", "Stock Analysis", "Chat with Advisor"])
    
    if page == "Dashboard":
        show_dashboard(portfolio_manager, analyzer)
    elif page == "Portfolio":
        show_portfolio(portfolio_manager)
    elif page == "Stock Analysis":
        show_stock_analysis(analyzer)
    elif page == "Chat with Advisor":
        show_chat(chatbot)

def show_dashboard(portfolio_manager, analyzer):
    st.header("Financial Dashboard")
    
    # Portfolio Summary Card
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        total_value, positions = portfolio_manager.get_portfolio_value()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", f"${total_value:,.2f}")
        with col2:
            daily_change = sum(p['gain_loss'] for p in positions)
            st.metric("Daily Change", f"${daily_change:,.2f}")
        with col3:
            if positions:
                best_performer = max(positions, key=lambda x: x['gain_loss_pct'])
                st.metric("Best Performer", f"{best_performer['symbol']} ({best_performer['gain_loss_pct']:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Market Overview
    with st.container():
        st.subheader("Market Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            spy_data = yf.download("SPY", period="1d")
            st.metric("S&P 500", f"${spy_data['Close'][-1]:.2f}", 
                     f"{((spy_data['Close'][-1] - spy_data['Open'][0]) / spy_data['Open'][0] * 100):.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            vix_data = yf.download("^VIX", period="1d")
            st.metric("VIX", f"{vix_data['Close'][-1]:.2f}", 
                     f"{((vix_data['Close'][-1] - vix_data['Open'][0]) / vix_data['Open'][0] * 100):.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

def show_portfolio(portfolio_manager):
    st.header("Portfolio Management")
    
    # Add New Position
    with st.expander("Add New Position"):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Stock Symbol").upper()
        with col2:
            shares = st.number_input("Number of Shares", min_value=0.0)
        with col3:
            price = st.number_input("Purchase Price", min_value=0.0)
        
        if st.button("Add Position"):
            if symbol and shares and price:
                portfolio_manager.add_position(symbol, shares, price)
                st.success(f"Added {shares} shares of {symbol} at ${price:.2f}")
    
    # Portfolio Table
    st.subheader("Current Holdings")
    total_value, positions = portfolio_manager.get_portfolio_value()
    
    if positions:
        df = pd.DataFrame(positions)
        df = df.round(2)
        st.dataframe(df.style.format({
            'purchase_price': '${:.2f}',
            'current_price': '${:.2f}',
            'position_value': '${:,.2f}',
            'gain_loss': '${:,.2f}',
            'gain_loss_pct': '{:.2f}%'
        }))
        
        # Portfolio Visualization
        fig = go.Figure(data=[go.Pie(
            labels=[p['symbol'] for p in positions],
            values=[p['position_value'] for p in positions],
            hole=.3
        )])
        fig.update_layout(title="Portfolio Allocation")
        st.plotly_chart(fig)
    else:
        st.info("No positions in portfolio. Add some positions to get started!")

[Previous show_stock_analysis and show_chat functions remain the same...]

if __name__ == "__main__":
    main()
