📈 Financial News Sentiment Dashboard

This Streamlit application provides real-time financial sentiment analysis on stock tickers using news articles fetched from NewsAPI.org and processed by the specialized FinBERT language model. It helps users visualize the sentiment trend (rolling average) and the overall distribution of positive, negative, and neutral headlines for any given public company.

✨ Key Features

This dashboard offers several crucial features for financial trend monitoring:

1. News Aggregation: It efficiently fetches up to 100 recent, English-language news articles related to a specified stock ticker (e.g., AAPL, TSLA) from NewsAPI.org.

2. Specialized Sentiment Analysis: The core analysis is powered by the FinBERT (Financial BERT) model, a state-of-the-art model fine-tuned specifically for financial text. This model assigns a score and a label (Positive, Neutral, Negative) to each article's headline and summary.

3. Trend Visualization: A time-series chart of the 5-Day Rolling Average Sentiment Score is displayed, enabling users to quickly identify shifts and momentum in market sentiment over time.

4. Distribution Breakdown: A clear Pie Chart shows the overall percentage of Positive, Neutral, and Negative headlines over the fetched period, giving a snapshot of the general market mood.

5. Secure Configuration: The application is built to use Streamlit Secrets for secure management of the NewsAPI key.

🛠️ Installation and Setup

1. Prerequisites

Before running the application, ensure you have Python 3.8+ installed and have obtained a valid API Key from NewsAPI.org.

2. Clone the Repository

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd financial-sentiment-dashboard


3. Install Dependencies

It's highly recommended to create and activate a virtual environment first. You can install all necessary packages—including streamlit, pandas, requests, plotly, and the transformers library for FinBERT—using the following command:

pip install streamlit pandas requests plotly transformers


4. Configure API Key (Crucial Security Step)

To run this application securely, you must configure your NewsAPI key using Streamlit's secrets management:

Create a folder named .streamlit in the root of your project directory.

Inside this folder, create a file named secrets.toml.

Add your NewsAPI key to this file using the exact variable name required by the application:

.streamlit/secrets.toml

NEWSAPI_API_KEY = "PASTE_YOUR_API_KEY_HERE"


5. Running the Application

Once your API key is configured, execute the application using Streamlit:

streamlit run streamlit_news_sentiment.py


This will launch the dashboard in your default web browser (usually at http://localhost:8501). Enter a stock ticker in the sidebar to begin the analysis.

⚙️ Technical Overview

The entire application is written in Python, utilizing the Streamlit framework for rapid user interface development. The data is acquired by making API calls to NewsAPI's /v2/everything endpoint. This data is then processed using Pandas for cleaning, aggregation, and calculating the crucial 5-day rolling average. For sentiment classification, the Hugging Face transformers library is leveraged to run the specialized FinBERT model. Finally, the visualizations, including the interactive line chart and pie chart, are powered by Plotly Express.