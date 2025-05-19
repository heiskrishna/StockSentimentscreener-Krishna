# Stock Analysis and Prediction App using Sentiment Analysis
# Developed by Krishna

import os
import requests
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
from textblob import TextBlob
from flask import Flask, render_template, request

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Helper Functions ---
def fetch_tweets(query, max_results=10):
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={max_results}&tweet.fields=created_at"
    response = requests.get(url, headers=headers)
    tweets = response.json().get("data", [])
    return [tweet["text"] for tweet in tweets]

def fetch_news(query, sources, page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&sources={sources}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [article["title"] for article in articles]

def clean_text(text):
    return text.replace('\n', ' ').strip()

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def aggregate_sentiment(headlines):
    sentiments = [get_sentiment(clean_text(h)) for h in headlines]
    return sum(sentiments) / len(sentiments) if sentiments else 0

def predict_stock_trend(sentiment_score):
    if sentiment_score > 0.1:
        return "Bullish"
    elif sentiment_score < -0.1:
        return "Bearish"
    else:
        return "Neutral"

def fetch_stock_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=30)
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

def create_stock_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Stock Price Trend', xaxis_title='Date', yaxis_title='Price (USD)')
    return fig.to_html(full_html=False)

# --- Web App Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    sentiment_score = None
    chart_html = None
    ticker = "AAPL"

    if request.method == "POST":
        ticker = request.form["ticker"]
        twitter_headlines = fetch_tweets(ticker)
        news_headlines = fetch_news(ticker, "business-standard,moneycontrol")
        all_headlines = twitter_headlines + news_headlines

        sentiment_score = aggregate_sentiment(all_headlines)
        prediction = predict_stock_trend(sentiment_score)

        stock_df = fetch_stock_data(ticker)
        chart_html = create_stock_chart(stock_df)

    return render_template("index.html", ticker=ticker, prediction=prediction, sentiment_score=sentiment_score, chart_html=chart_html)

if __name__ == "__main__":
    app.run(debug=True)
