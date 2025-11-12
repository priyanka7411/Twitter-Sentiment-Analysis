import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)







import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import re
import string
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1DA1F2;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize VADER sentiment analyzer
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess tweet text"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags symbols (keep the words)
    text = re.sub(r'[@#]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Sentiment analysis with TextBlob
def get_textblob_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if not text:
        return 'Neutral', 0.0
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, polarity

# Sentiment analysis with VADER
def get_vader_sentiment(text):
    """Analyze sentiment using VADER"""
    if not text:
        return 'Neutral', 0.0
    
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, compound

# Generate word cloud
def generate_wordcloud(text_data):
    """Generate word cloud from text"""
    text = ' '.join(text_data)
    
    if not text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    return wordcloud

# Main application
def main():
    # Header
    st.title("🐦 Twitter Sentiment Analysis Dashboard")
    st.markdown("### Analyze sentiments from tweets using NLP")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        analysis_method = st.selectbox(
            "Select Sentiment Analysis Method",
            ["VADER", "TextBlob", "Both (Comparison)"]
        )
        
        st.markdown("---")
        
        st.markdown("### 📖 About This App")
        st.info("""
        **Features:**
        - Single tweet analysis
        - Bulk CSV upload
        - Visual analytics
        - Word clouds
        - Sentiment distribution
        
        **Methods:**
        - **VADER**: Best for social media
        - **TextBlob**: General purpose
        """)
        
        st.markdown("---")
        st.markdown("### 👩‍💻 Developer")
        st.markdown("**Priyanka**")
        st.markdown("BCA Graduate 2024")
        st.markdown("Data Science Portfolio")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Tweet", "📁 Bulk Analysis", "📊 Sample Dataset"])
    
    # TAB 1: Single Tweet Analysis
    with tab1:
        st.subheader("Analyze Individual Tweet")
        
        tweet_input = st.text_area(
            "Enter tweet text:",
            height=120,
            placeholder="Example: I love this product! It's amazing and works perfectly! 😊",
            help="Paste any tweet or text you want to analyze"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if analyze_btn and tweet_input.strip():
            with st.spinner("Analyzing sentiment..."):
                cleaned_text = preprocess_text(tweet_input)
                
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Original and cleaned text
                with st.expander("📄 Text Preprocessing", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Original Text:**")
                        st.text_area("", tweet_input, height=100, disabled=True, label_visibility="collapsed")
                    with col_b:
                        st.markdown("**Cleaned Text:**")
                        st.text_area("", cleaned_text, height=100, disabled=True, label_visibility="collapsed")
                
                # Sentiment results
                st.markdown("### 🎯 Sentiment Scores")
                
                if analysis_method == "VADER" or analysis_method == "Both (Comparison)":
                    vader_sentiment, vader_score = get_vader_sentiment(cleaned_text)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("VADER Sentiment", vader_sentiment)
                    with col2:
                        st.metric("Compound Score", f"{vader_score:.3f}")
                    with col3:
                        confidence = abs(vader_score) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Visual representation
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=vader_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "VADER Sentiment Score"},
                        delta={'reference': 0},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-1, -0.05], 'color': "lightcoral"},
                                {'range': [-0.05, 0.05], 'color': "lightyellow"},
                                {'range': [0.05, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                if analysis_method == "TextBlob" or analysis_method == "Both (Comparison)":
                    tb_sentiment, tb_score = get_textblob_sentiment(cleaned_text)
                    
                    if analysis_method == "Both (Comparison)":
                        st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("TextBlob Sentiment", tb_sentiment)
                    with col2:
                        st.metric("Polarity Score", f"{tb_score:.3f}")
                    with col3:
                        confidence = abs(tb_score) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Visual representation
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=tb_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "TextBlob Sentiment Score"},
                        delta={'reference': 0},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [-1, -0.1], 'color': "lightcoral"},
                                {'range': [-0.1, 0.1], 'color': "lightyellow"},
                                {'range': [0.1, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Comparison if both selected
                if analysis_method == "Both (Comparison)":
                    st.markdown("---")
                    st.markdown("### 📊 Method Comparison")
                    
                    comparison_df = pd.DataFrame({
                        'Method': ['VADER', 'TextBlob'],
                        'Sentiment': [vader_sentiment, tb_sentiment],
                        'Score': [vader_score, tb_score]
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        elif analyze_btn and not tweet_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
    
    # TAB 2: Bulk Analysis
    with tab2:
        st.subheader("Upload CSV File for Bulk Analysis")
        
        st.info("📋 **CSV Format Required:** Your file should have a column named 'text' or 'tweet' containing the tweets.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with tweets"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
                
                # Detect text column
                text_column = None
                for col in ['text', 'tweet', 'Tweet', 'Text', 'content', 'Content']:
                    if col in df.columns:
                        text_column = col
                        break
                
                if text_column is None:
                    st.error("❌ No 'text' or 'tweet' column found. Please ensure your CSV has one of these columns.")
                else:
                    st.info(f"📌 Using column: **{text_column}**")
                    
                    # Show preview
                    with st.expander("👀 Preview Data", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("🚀 Start Analysis", type="primary"):
                        with st.spinner("Analyzing all tweets... This may take a moment..."):
                            # Clean texts
                            df['cleaned_text'] = df[text_column].apply(preprocess_text)
                            
                            # Apply sentiment analysis
                            if analysis_method == "VADER" or analysis_method == "Both (Comparison)":
                                vader_results = df['cleaned_text'].apply(lambda x: get_vader_sentiment(x))
                                df['vader_sentiment'] = vader_results.apply(lambda x: x[0])
                                df['vader_score'] = vader_results.apply(lambda x: x[1])
                            
                            if analysis_method == "TextBlob" or analysis_method == "Both (Comparison)":
                                tb_results = df['cleaned_text'].apply(lambda x: get_textblob_sentiment(x))
                                df['textblob_sentiment'] = tb_results.apply(lambda x: x[0])
                                df['textblob_score'] = tb_results.apply(lambda x: x[1])
                            
                            st.success("✅ Analysis complete!")
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📊 Analysis Results")
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            sentiment_col = 'vader_sentiment' if 'vader_sentiment' in df.columns else 'textblob_sentiment'
                            
                            with col1:
                                st.metric("Total Tweets", len(df))
                            with col2:
                                positive = len(df[df[sentiment_col] == 'Positive'])
                                st.metric("Positive", positive, delta=f"{positive/len(df)*100:.1f}%")
                            with col3:
                                neutral = len(df[df[sentiment_col] == 'Neutral'])
                                st.metric("Neutral", neutral, delta=f"{neutral/len(df)*100:.1f}%")
                            with col4:
                                negative = len(df[df[sentiment_col] == 'Negative'])
                                st.metric("Negative", negative, delta=f"{negative/len(df)*100:.1f}%")
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Pie chart
                                sentiment_counts = df[sentiment_col].value_counts()
                                fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="Sentiment Distribution",
                                    color=sentiment_counts.index,
                                    color_discrete_map={
                                        'Positive': '#00CC96',
                                        'Neutral': '#FFA15A',
                                        'Negative': '#EF553B'
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Bar chart
                                fig = px.bar(
                                    x=sentiment_counts.index,
                                    y=sentiment_counts.values,
                                    title="Sentiment Count",
                                    labels={'x': 'Sentiment', 'y': 'Count'},
                                    color=sentiment_counts.index,
                                    color_discrete_map={
                                        'Positive': '#00CC96',
                                        'Neutral': '#FFA15A',
                                        'Negative': '#EF553B'
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Word cloud
                            st.markdown("### ☁️ Word Cloud")
                            
                            sentiment_filter = st.selectbox(
                                "Select sentiment for word cloud:",
                                ['All', 'Positive', 'Negative', 'Neutral']
                            )
                            
                            if sentiment_filter == 'All':
                                text_for_cloud = df['cleaned_text'].tolist()
                            else:
                                text_for_cloud = df[df[sentiment_col] == sentiment_filter]['cleaned_text'].tolist()
                            
                            if text_for_cloud:
                                wordcloud = generate_wordcloud(text_for_cloud)
                                if wordcloud:
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                            
                            # Show detailed results
                            with st.expander("📋 View Detailed Results"):
                                st.dataframe(df, use_container_width=True)
                            
                            # Download results
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # TAB 3: Sample Dataset
    with tab3:
        st.subheader("Try with Sample Dataset")
        
        st.info("📌 Click below to generate and analyze sample tweets!")
        
        if st.button("🎲 Generate Sample Data", type="primary"):
            # Sample tweets
            sample_tweets = [
                "I absolutely love this product! Best purchase ever! 😊",
                "This is terrible. Worst experience of my life.",
                "It's okay, nothing special about it.",
                "Amazing customer service! They went above and beyond!",
                "I'm so disappointed with this. Complete waste of money.",
                "The quality is decent for the price.",
                "Fantastic! Exceeded all my expectations!",
                "Not worth it at all. Very unhappy with this purchase.",
                "It works as described. No complaints.",
                "Brilliant! I recommend this to everyone!",
                "This is horrible. Would not recommend to anyone.",
                "Average product. Does the job.",
                "Outstanding quality and fast delivery!",
                "Poor quality and bad customer support.",
                "It's fine. Nothing to write home about."
            ]
            
            sample_df = pd.DataFrame({'text': sample_tweets})
            
            # Analyze
            sample_df['cleaned_text'] = sample_df['text'].apply(preprocess_text)
            
            vader_results = sample_df['cleaned_text'].apply(lambda x: get_vader_sentiment(x))
            sample_df['sentiment'] = vader_results.apply(lambda x: x[0])
            sample_df['score'] = vader_results.apply(lambda x: x[1])
            
            st.success("✅ Sample data generated and analyzed!")
            
            # Show results
            st.dataframe(sample_df[['text', 'sentiment', 'score']], use_container_width=True, hide_index=True)
            
            # Visualize
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = sample_df['sentiment'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sample Data Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#00CC96',
                        'Neutral': '#FFA15A',
                        'Negative': '#EF553B'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    title="Sentiment Counts",
                    labels={'x': 'Sentiment', 'y': 'Count'},
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#00CC96',
                        'Neutral': '#FFA15A',
                        'Negative': '#EF553B'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()