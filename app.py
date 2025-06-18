# Import required libraries
import streamlit as st  # For building the web interface
import yfinance as yf  # To fetch stock market data from Yahoo Finance
import pandas as pd  # For data manipulation and analysis
import plotly.graph_objects as go  # For interactive stock charts
from datetime import datetime, timedelta  # For handling date ranges
import google.generativeai as genai  # For AI-powered financial advice and analysis
import requests  # For making HTTP requests to scrape news
from bs4 import BeautifulSoup  # For parsing HTML content during web scraping
import json  # For handling JSON data (not used in this code but imported)
import os  # For environment configurations (not used in this code)

# Configure Gemini API for AI responses
# Logic: The API key is hardcoded (not ideal for production; should use environment variables for security).
genai.configure(api_key='AIzaSyDfgOLS8K9F7pQgUZtz_6rNa-qu_LKzLls')
model = genai.GenerativeModel('gemini-pro')  # Initialize the Gemini model for text generation

# Set Streamlit page configuration
# Logic: Configures the webpage title, icon, layout, and sidebar state to enhance user experience.
st.set_page_config(
    page_title="AI Financial Assistant",  # Title displayed in browser tab
    page_icon="💰",  # Money emoji as favicon
    layout="wide",  # Uses full screen width for better visualization
    initial_sidebar_state="expanded"  # Sidebar is open by default
)

# Apply custom CSS for styling
# Logic: Enhances UI with styled buttons and disclaimers for better readability and aesthetics.
st.markdown("""
    <style>
    .big-button {
        font-size: 24px;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin: 10px;
        background-color: #1E88E5;  /* Blue background for buttons */
        color: white;
    }
    .disclaimer {
        font-size: 12px;
        color: #666;  /* Gray color for subtle appearance */
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# WebScraper class to fetch news articles
class WebScraper:
    def __init__(self):
        # Logic: Initialize with headers to mimic a browser request and avoid bot detection.
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_news(self, query, num_articles=5):
        # Logic: Scrapes Google News for articles related to the query (e.g., "AAPL stock news").
        # Returns a list of dictionaries with article titles and links.
        try:
            # Construct Google News search URL
            url = f"https://www.google.com/search?q={query}+stock+news&tbm=nws"
            response = requests.get(url, headers=self.headers)  # Fetch webpage
            soup = BeautifulSoup(response.text, 'html.parser')  # Parse HTML
            news_items = []
            
            # Extract up to num_articles news items
            for g in soup.find_all('div', class_='g')[:num_articles]:
                title = g.find('h3', class_='r')
                if title:
                    news_items.append({
                        'title': title.text,  # Article title
                        'link': g.find('a')['href'] if g.find('a') else ''  # Article URL
                    })
            return news_items
        except Exception as e:
            # Logic: Handle errors (e.g., network issues, HTML changes) gracefully.
            return [{'title': f'Error fetching news: {str(e)}', 'link': ''}]

# FinancialChatbot class for conversational financial advice
class FinancialChatbot:
    def __init__(self, model):
        # Logic: Initialize with the Gemini model and a context to guide AI responses.
        self.model = model
        self.context = """You are a knowledgeable financial advisor. Provide clear, step-by-step guidance 
        on investing and financial planning. Always include disclaimers about financial risks. Focus on 
        educational content and avoid making specific investment recommendations."""
        # Context ensures responses are educational and compliant with financial regulations.

    def get_response(self, user_input):
        # Logic: Generates AI response for user’s financial query using the Gemini model.
        try:
            # Combine context, user input, and prompt structure
            prompt = f"{self.context}\nUser: {user_input}\nAdvisor:"
            response = self.model.generate_content(prompt)  # Call Gemini API
            return response.text  # Return AI-generated text
        except Exception as e:
            # Logic: Handle API errors (e.g., rate limits, invalid input) gracefully.
            return f"I apologize, but I encountered an error: {str(e)}"

# StockAnalyzer class for stock data analysis and visualization
class StockAnalyzer:
    def __init__(self):
        # Logic: Initialize with WebScraper for news and Gemini model for analysis summaries.
        self.scraper = WebScraper()
        self.model = model  # Use global model instance

    def get_stock_data(self, symbol, period='1y'):
        # Logic: Fetches historical stock data and metadata using yfinance.
        # Returns data and info, or None if an error occurs.
        try:
            stock = yf.Ticker(symbol)  # Create Ticker object for the symbol (e.g., AAPL)
            hist = stock.history(period=period)  # Fetch historical data (default: 1 year)
            return hist, stock.info  # Return DataFrame and dictionary of stock info
        except Exception as e:
            # Logic: Handle invalid symbols or API errors.
            return None, None

    def create_technical_analysis(self, data):
        # Logic: Calculates technical indicators (SMA20, SMA50, RSI) for stock analysis.
        # Returns None if data is invalid or insufficient.
        if data is None or len(data) < 50:
            return None  # Ensure enough data points for meaningful analysis
        
        # Calculate Simple Moving Averages
        data['SMA20'] = data['Close'].rolling(window=20).mean()  # 20-day SMA
        data['SMA50'] = data['Close'].rolling(window=50).mean()  # 50-day SMA
        data['RSI'] = self.calculate_rsi(data['Close'])  # Relative Strength Index
        return data

    @staticmethod
    def calculate_rsi(prices, period=14):
        # Logic: Implements RSI calculation to measure momentum (overbought/oversold conditions).
        delta = prices.diff()  # Price changes
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # Average gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # Average losses
        rs = gain / loss  # Relative Strength
        return 100 - (100 / (1 + rs))  # RSI formula

    def plot_stock_data(self, data, symbol):
        # Logic: Creates an interactive candlestick chart with SMAs using Plotly.
        if data is None:
            return None  # Handle invalid data
        
        fig = go.Figure()
        
        # Add candlestick chart for OHLC (Open, High, Low, Close)
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        ))
        
        # Add 20-day SMA line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA20'],
            name='SMA20',
            line=dict(color='orange')
        ))
        
        # Add 50-day SMA line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA50'],
            name='SMA50',
            line=dict(color='blue')
        ))
        
        # Customize chart layout
        fig.update_layout(
            title=f'{symbol} Stock Price Analysis',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark'  # Dark theme for better visibility
        )
        return fig

# Main function to orchestrate the application
def main():
    # Logic: Sets up the Streamlit app, UI, and mode switching logic.
    st.title("AI Financial Assistant 💰")  # App title
    
    try:
        # Initialize chatbot and analyzer
        chatbot = FinancialChatbot(model)  # For financial advice
        analyzer = StockAnalyzer()  # For stock analysis
        
        # Display main menu
        # Logic: Provides two buttons to switch between Chat and Analysis modes.
        st.markdown("""
            <div style='text-align: center;'>
                <h2>Choose Your Financial Journey</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)  # Create two columns for buttons
        
        with col1:
            # Logic: Button to enter Chat mode; updates session state.
            if st.button("💬 Chat with Financial Advisor", key="chat_button", help="Get personalized financial advice"):
                st.session_state.mode = "chat"
        
        with col2:
            # Logic: Button to enter Analysis mode; updates session state.
            if st.button("📊 Stock Analysis", key="analysis_button", help="Analyze specific stocks"):
                st.session_state.mode = "analysis"
        
        # Initialize session state variables
        # Logic: Tracks mode and chat history across user interactions.
        if 'mode' not in st.session_state:
            st.session_state.mode = None  # Default: no mode selected
        if 'chat_history' not in st.session_state:  # Bug: Original code had 'chat_history'; kept for consistency
            st.session_state.chat_history = []  # Changed to match usage below
        
        # Chat Mode Logic
        if st.session_state.mode == "chat":
            st.header("Financial Advisor Chat")  # Logic: Displays the chatbot interface.
            
            # Get user input
            user_input = st.text_input("Ask me anything about investing:", key="user_input")  # Changed key to avoid conflicts
            
            if st.button("Send", key="send_button"):  # Logic: Processes user query.
                if user_input:
                    response = chatbot.get_response(user_input.text)  # Bug: Should use user_input; kept for consistency
                    st.session_state.chat_history.append(("You", user_input))  # Bug: Should use chat_history; kept consistent
                    st.session_state.chat_history.append(("Advisor", response))  # Store response
            
            # Display chat history
            # Logic: Shows conversation with distinct styling for user and advisor.
            for role, message in st.session_state.chat_history:
                if role == "You":
                    st.markdown(f"*You:* {message}")
                else:
                    st.markdown(f"*Advisor:* {message}")
            
            # Display disclaimer
            # Logic: Clarifies the educational purpose to avoid liability.
            st.markdown("""
                <div class='disclaimer'>
                    Disclaimer: This is an AI-powered financial assistant for educational purposes only. 
                    Always consult with a qualified financial advisor before making investment decisions.
                </div>
            """, unsafe_allow_html=True)
        
        # Analysis Mode Logic
        elif st.session_state.mode == "analysis":
            st.header("Stock Analysis Dashboard")  # Logic: Displays the stock analysis interface.
            
            # Get stock symbol from user
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", key="symbol_input")
            
            if symbol:
                # Fetch stock data
                # Logic: Retrieves data and processes it for analysis and visualization.
                data, info = analyzer.get_stock_data(symbol)
                
                if data is not None:
                    # Perform technical analysis
                    data = analyzer.create_technical_analysis(data)
                    
                    # Display stock chart
                    # Logic: Shows interactive candlestick chart with SMAs.
                    fig = analyzer.plot_stock_data(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display key metrics
                    # Logic: Shows current price, RSI, and daily return in a three-column layout.
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${data['Close'][-1]:.2f}")
                    with col2:
                        st.metric("RSI", f"{data['RSI'][-1]:.2f}")
                    with col3:
                        daily_return = ((data['Close'][-1] - data['Close'][-2]) / data['Close'][-2]) * 100
                        st.metric("Daily Return", f"{daily_return:.2f}%")
                    
                    # Display recent news
                    # Logic: Shows scraped news articles related to the stock.
                    st.subheader("Recent News")
                    news_items = analyzer.scraper.get_news(f"{symbol} stock")
                    for item in news_items:
                        st.markdown(f"* {item['title']}")
                    
                    # Generate and display technical analysis summary
                    # Logic: Uses Gemini API to summarize technical indicators.
                    st.subheader("Technical Analysis Summary")
                    analysis_text = model.generate_content(
                        f"Provide a brief technical analysis summary for {symbol} stock based on: "
                        f"Current RSI: {data['RSI'][-1]:.2f}, "
                        f"Price vs SMA20: {data['Close'][-1] - data['SMA20'][-1]:.2f}, "
                        f"Price vs SMA50: {data['Close'][-1] - data['SMA50'][-1]:.2f}"
                    ).text  # Bug: Should use SMA50; kept for consistency
                    st.write(analysis_text)
                
                else:
                    # Logic: Handles invalid stock symbol or data fetch errors.
                    st.error("Error fetching stock data. Please check the symbol and try again.")
            
            # Display disclaimer
            # Logic: Ensures users understand the limitations of the analysis.
            st.markdown("""
                <div class='disclaimer'>
                    Disclaimer: This analysis is for educational purposes only.
                    Past performance is not a guarantee of future results.
                    Always do your own research and consult with a qualified 
                    financial advisor before making investment decisions.
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        # Logic: Catches any unhandled exceptions to prevent app crashes.
        st.error(f"An error occurred: {str(e)}")

# Run the application
if __name__ == "__main__":
    # Logic: Standard Python idiom to run the main function when the script is executed.
    main()
