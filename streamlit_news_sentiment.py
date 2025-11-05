import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from transformers import pipeline
import numpy as np

# --- 1. CONFIGURATION & CACHING (Phase 1) ---

# Set Streamlit page layout to wide for a professional look
st.set_page_config(layout="wide", page_title="Financial Sentiment Dashboard")

# --- Security & Caching Best Practice ---
try:
    # Attempt to load API Key from Streamlit Secrets or Environment Variables
    # NOTE: Key variable changed to NEWSAPI_KEY
    NEWSAPI_KEY = os.environ.get("NEWSAPI_API_KEY") or st.secrets["NEWSAPI_API_KEY"]
except (KeyError, AttributeError):
    # !!! IMPORTANT: Replace this placeholder with your actual key for local testing !!!
    NEWSAPI_KEY = "YOUR_API_KEY"

# 1.1 Model Caching: Load the heavy FinBERT model only ONCE
@st.cache_resource(show_spinner="Loading FinBERT Model (This happens only once)...")
def load_finbert_model():
    """Loads the specialized FinBERT model for financial sentiment."""
    # ProsusAI/finbert is the preferred model for finance sentiment
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 1.2 Data Caching: Cache the API call for 1 hour (3600 seconds) to prevent rate limits
@st.cache_data(ttl=3600, show_spinner="Fetching and preparing news data...")
def fetch_news_data(ticker, api_key):
    """
    Fetches news articles related to the ticker from NewsAPI using the 'Everything' endpoint
    and returns a DataFrame.
    """
    
    # NewsAPI 'Everything' Endpoint for relevant financial news
    # Search query focuses on the ticker and related financial terms
    query = f"{ticker} stock OR {ticker} earnings OR {ticker} finance"
    # Fetch 100 English articles sorted by date
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize=100&language=en&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises HTTPError for bad status codes
        data = response.json()
        
        articles = data.get('articles', [])

        if not articles:
             st.warning(f"Found 0 articles for {ticker.upper()}. Try a different ticker or check your NewsAPI key/rate limits.")
             return pd.DataFrame()

        # Map NewsAPI fields to the expected DataFrame structure
        parsed_data = []
        for article in articles:
            # Combine title and description/content for richer sentiment context (used for the 'summary' column)
            text_for_analysis = f"{article.get('title', '')}. {article.get('description', '') or article.get('content', '')}"
            
            parsed_data.append({
                'date': article.get('publishedAt'),
                # Ticker is derived from the user input since NewsAPI doesn't provide it per article
                'ticker': ticker, 
                'headline': article.get('title'),
                'summary': text_for_analysis
            })
            
        df = pd.DataFrame(parsed_data)
        
        # Convert date column to datetime objects
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop rows where date or headline/summary is missing/invalid
        df.dropna(subset=['date', 'headline', 'summary'], inplace=True)
        
        # Remove duplicates based on the headline
        df.drop_duplicates(subset=['headline'], inplace=True)
        
        return df.sort_values(by='date', ascending=False).reset_index(drop=True)
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            st.error("Error 401: Unauthorized. Please check your **NewsAPI Key**. It might be invalid or expired.")
        elif response.status_code == 429:
            st.error("Error 429: Rate Limit Exceeded. You have made too many requests recently. Wait and try again.")
        else:
            st.error(f"Error fetching data for {ticker.upper()}. NewsAPI Error: {e}")
        return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        st.error(f"A network error occurred while reaching NewsAPI. Error: {e}")
        return pd.DataFrame()

# --- 2. SENTIMENT & METRICS PROCESSING (Phase 2) ---

def apply_sentiment_to_df(df, finbert_model):
    """2.1 Applies FinBERT sentiment analysis to the 'summary' column in a batched manner."""
    if df.empty:
        return df

    with st.spinner("Applying FinBERT Sentiment Analysis..."):
        # 1. Extract summaries (which contain title + description) and run batch inference
        summaries = df['summary'].tolist()
        # Use batch processing for efficiency (adjust batch_size based on system memory)
        results = finbert_model(summaries, batch_size=64) 
        
        # 2. Extract labels and scores
        df['sentiment_label'] = [r['label'] for r in results]
        df['raw_score'] = [r['score'] for r in results]

        # 3. Map categorical labels to numerical scores (+1, 0, -1) and apply raw score
        label_to_sign = {'positive': 1, 'neutral': 0, 'negative': -1}
        # The final numerical sentiment is the signed raw score
        df['numerical_sentiment'] = df.apply(
            lambda row: label_to_sign.get(row['sentiment_label'].lower(), 0) * row['raw_score'],
            axis=1
        )
        
    return df

def calculate_metrics(df):
    """2.2 Aggregates data for time-series and calculates rolling metrics."""
    if df.empty:
        # Return empty DataFrames to avoid errors
        return pd.DataFrame({'date': [], 'daily_avg_sentiment': [], 'rolling_avg_sentiment': []}), \
               pd.DataFrame({'Sentiment': [], 'Count': []})
    
    # 1. Aggregate Daily Sentiment
    df_ts = df.set_index('date')
    # Calculate the mean of the numerical_sentiment for all articles on a given day
    daily_sentiment = df_ts.groupby(df_ts.index.date)['numerical_sentiment'].mean().reset_index()
    daily_sentiment.columns = ['date', 'daily_avg_sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

    # 2. Calculate Rolling Average (The key trend feature)
    window_size = 5 # Use a 5-day window for robust trend visibility
    daily_sentiment['rolling_avg_sentiment'] = (
        daily_sentiment['daily_avg_sentiment']
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    
    # 3. Calculate Overall Distribution for the pie chart
    sentiment_counts = df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    return daily_sentiment, sentiment_counts

# --- 3. STREAMLIT UI BUILDING (Phase 3) ---

def build_dashboard_ui(daily_df, counts_df, news_df, ticker):
    """Renders the charts and data in the Streamlit UI."""
    st.title(f"📈 Financial News Sentiment Dashboard: {ticker.upper()}")
    st.markdown("---")
    
    # 3.2 Use two columns for the main charts
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Sentiment Trend Analysis (5-Day Rolling Average)")
        
        # Plotly Line Chart for Rolling Average
        fig_line = px.line(
            daily_df, 
            x='date', 
            y='rolling_avg_sentiment',
            title=f"Sentiment Trend for {ticker.upper()}",
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#4c78a8'] # Blue/Standard color
        )
        # Add the neutral baseline
        fig_line.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral Baseline", annotation_position="bottom right") 
        fig_line.update_layout(yaxis_range=[-1.0, 1.0])
        fig_line.update_xaxes(title_text="Date")
        fig_line.update_yaxes(title_text="Rolling Sentiment Score")
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.subheader("Headline Distribution")
        # Plotly Pie Chart for Distribution
        fig_pie = px.pie(
            counts_df, 
            values='Count', 
            names='Sentiment', 
            title="Overall Breakdown",
            color='Sentiment',
            color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'#94a3b8'} # Slate/Gray for neutral
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Processed Headlines (Validation)")
    st.caption(f"Showing the top {min(20, len(news_df))} most recent processed articles with FinBERT scores.")
    
    # Display the processed data for validation
    st.dataframe(
        news_df[['date', 'headline', 'sentiment_label', 'numerical_sentiment']].head(20), 
        use_container_width=True,
        column_config={
            "date": st.column_config.DatetimeColumn("Date", format="YYYY/MM/DD HH:mm"),
            "headline": st.column_config.TextColumn("Headline"),
            "sentiment_label": st.column_config.TextColumn("Label"),
            "numerical_sentiment": st.column_config.ProgressColumn(
                "Final Score (-1 to +1)", min_value=-1, max_value=1, format="%.3f"
            )
        }
    )

# --- MAIN APPLICATION LOGIC (Phase 4) ---

def main():
    # 3.1 Sidebar for User Input
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL)", value="AAPL").upper().strip()
    
    # Check for the NewsAPI Key
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        st.error("🚨 API Key Missing: Please replace 'YOUR_NEWSAPI_KEY' in the code or set the NEWSAPI_API_KEY environment variable/secret.")
        return
        
    if not ticker:
        st.info("Please enter a stock ticker in the sidebar to begin the analysis.")
        return

    # 4.2 Step 1: Data Acquisition
    # Pass the updated key variable
    raw_news_df = fetch_news_data(ticker, NEWSAPI_KEY)
    
    if raw_news_df.empty:
        # Error message is handled inside fetch_news_data
        return

    # 4.3 Step 2: Load Model & Apply Sentiment
    finbert_model = load_finbert_model()
    scored_news_df = apply_sentiment_to_df(raw_news_df, finbert_model)
    
    # 4.4 Step 3: Calculate Metrics
    daily_metrics_df, counts_df = calculate_metrics(scored_news_df)
    
    if daily_metrics_df.empty:
        st.warning("Not enough data to calculate time-series metrics.")
        return
        
    # 4.5 Step 4: Build UI
    build_dashboard_ui(daily_metrics_df, counts_df, scored_news_df, ticker)

# 4.6 Run Entry Point
if __name__ == "__main__":
    main()
