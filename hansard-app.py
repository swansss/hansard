import streamlit as st
import requests
import pandas as pd
import altair as alt
import datetime
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import time
import random
import re

st.set_page_config(
    page_title="Parliament AI Tracker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Parliamentary AI Mentions Tracker")
st.markdown("""
This app tracks mentions of AI-related terms in the UK Parliament using the Hansard API.
Data focuses on the current parliamentary session, with recent mentions displayed first.
""")

# Define AI-related terms to track
DEFAULT_TERMS = [
    "artificial intelligence", 
    "google deepmind", 
    "deepmind", 
    "alphafold", 
    "large language model"
]

ADDITIONAL_TERMS = [
    "machine learning",
    "generative AI",
    "AI regulation",
    "AI safety",
    "AI ethics",
    "foundation models",
    "neural networks",
    "AI policy",
    "artificial general intelligence",
    "AGI",
    "transformer models"
]

# Sidebar for filters and options
st.sidebar.header("Filters and Options")

# Date range selector
today = datetime.date.today()
date_from = st.sidebar.date_input("From date", today - datetime.timedelta(days=365))
date_to = st.sidebar.date_input("To date", today)

# House selection
house_options = ["Both", "Commons", "Lords"]
selected_house = st.sidebar.selectbox("Select House", house_options)

# Term selection
with st.sidebar.expander("Search Terms", expanded=False):
    selected_terms = []
    for term in DEFAULT_TERMS:
        if st.checkbox(term, value=True, key=f"term_{term}"):
            selected_terms.append(term)
    
    st.markdown("#### Additional Terms")
    for term in ADDITIONAL_TERMS:
        if st.checkbox(term, value=False, key=f"term_{term}"):
            selected_terms.append(term)
    
    custom_term = st.text_input("Add custom term")
    if custom_term and st.button("Add Term"):
        selected_terms.append(custom_term.lower())
        st.success(f"Added: {custom_term}")

if not selected_terms:
    st.warning("Please select at least one term to track")
    selected_terms = DEFAULT_TERMS.copy()

# Function to query the Hansard API
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hansard_data(term, house, date_from, date_to):
    """Fetch data from Hansard API for a specific term."""
    base_url = "https://hansard-api.parliament.uk/search"
    
    # Convert dates to required format (YYYY-MM-DD)
    from_date_str = date_from.strftime("%Y-%m-%d")
    to_date_str = date_to.strftime("%Y-%m-%d")
    
    # Map house selection to API parameter
    house_param = ""
    if house == "Commons":
        house_param = "Commons"
    elif house == "Lords":
        house_param = "Lords"
    
    params = {
        "q": term,
        "house": house_param,
        "start": from_date_str,
        "end": to_date_str,
        "rows": 100  # Maximum rows per request
    }
    
    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Fetching data for term: {term}... (Attempt {attempt+1}/{max_retries})"):
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                return data
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                st.error(f"Failed to fetch data after {max_retries} attempts: {e}")
                # Mock data for development/demonstration if API is inaccessible
                if st.checkbox("Use sample data for development", value=False):
                    return generate_sample_data(term)
                return {"items": []}

# Function to process and combine data from multiple terms
def process_hansard_data(terms, house, date_from, date_to):
    """Process and combine data from multiple search terms."""
    all_results = []
    
    with st.spinner("Fetching and processing data..."):
        progress_bar = st.progress(0)
        
        for i, term in enumerate(terms):
            data = fetch_hansard_data(term, house, date_from, date_to)
            
            if "items" in data:
                for item in data["items"]:
                    # Add the search term that found this result
                    item["search_term"] = term
                    all_results.append(item)
            
            # Update progress
            progress_bar.progress((i + 1) / len(terms))
            time.sleep(0.1)  # Small delay to prevent rate limiting
        
        progress_bar.empty()
    
    # Convert to DataFrame for easier processing
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Clean and process the data
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            
            # Remove duplicates based on content ID
            if 'id' in df.columns:
                df = df.drop_duplicates(subset=['id'])
            
            return df
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# Function to extract MP/Lord names and count mentions
def count_speaker_mentions(df):
    """Count mentions by each speaker and return sorted DataFrame."""
    if df.empty or 'member' not in df.columns:
        return pd.DataFrame(columns=['name', 'house', 'count'])
    
    # Extract member data and count
    speaker_counts = Counter()
    speaker_houses = {}
    
    for _, row in df.iterrows():
        if pd.notna(row.get('member')) and isinstance(row['member'], dict):
            name = row['member'].get('name')
            house = row['member'].get('house')
            
            if name:
                speaker_counts[name] += 1
                if house:
                    speaker_houses[name] = house
    
    # Convert to DataFrame
    speakers_df = pd.DataFrame({
        'name': list(speaker_counts.keys()),
        'count': list(speaker_counts.values())
    })
    
    # Add house information
    speakers_df['house'] = speakers_df['name'].map(speaker_houses)
    
    # Sort by count (descending)
    speakers_df = speakers_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    return speakers_df

# Function to create a trend over time chart
def create_trend_chart(df):
    """Create a line chart showing trends over time."""
    if df.empty or 'date' not in df.columns:
        return None
    
    # Resample data by month and count
    df['month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby('month').size().reset_index(name='count')
    monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()
    
    # Create chart
    chart = alt.Chart(monthly_counts).mark_line(point=True).encode(
        x=alt.X('month:T', title='Month'),
        y=alt.Y('count:Q', title='Number of Mentions'),
        tooltip=['month:T', 'count:Q']
    ).properties(
        title='AI-related Mentions Over Time',
        width=700,
        height=400
    ).interactive()
    
    return chart

# Function to create word cloud of context
def create_word_cloud(df):
    """Create a word cloud from the content of mentions."""
    if df.empty or 'description' not in df.columns:
        return None
    
    # Combine all descriptions
    text = ' '.join(df['description'].dropna().astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        collocations=False
    ).generate(text)
    
    # Display word cloud using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# Function to create term frequency comparison
def create_term_frequency_chart(df):
    """Create a bar chart comparing frequencies of different terms."""
    if df.empty or 'search_term' not in df.columns:
        return None
    
    # Count occurrences of each search term
    term_counts = df['search_term'].value_counts().reset_index()
    term_counts.columns = ['term', 'count']
    
    # Create chart
    chart = alt.Chart(term_counts).mark_bar().encode(
        x=alt.X('count:Q', title='Number of Mentions'),
        y=alt.Y('term:N', title='Term', sort='-x'),
        color=alt.Color('term:N', legend=None),
        tooltip=['term:N', 'count:Q']
    ).properties(
        title='Frequency of AI-related Terms',
        width=700,
        height=400
    )
    
    return chart

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["MP/Lord Rankings", "Trends Over Time", "Term Comparison", "Recent Mentions"])

# Function to generate sample data when API is unavailable
def generate_sample_data(term):
    """Generate mock data for development and testing."""
    st.warning("Using mock data for demonstration purposes. Real API data may differ.")
    
    # Create a list of sample MPs and Lords
    commons_members = [
        "John Smith MP", "Sarah Johnson MP", "David Williams MP", "Emma Brown MP",
        "Michael Wilson MP", "Olivia Jones MP", "Robert Taylor MP", "Sophia Davies MP"
    ]
    
    lords_members = [
        "Lord Anderson", "Baroness Mitchell", "Lord Roberts", "Baroness White",
        "Lord Thomas", "Baroness Clark", "Lord Martin", "Baroness Lewis"
    ]
    
    # Create sample items
    items = []
    today = datetime.datetime.now()
    
    for i in range(50):  # Generate 50 sample items
        # Decide if it's Commons or Lords
        is_commons = random.choice([True, False])
        
        # Select a random member
        if is_commons:
            member_name = random.choice(commons_members)
            house = "Commons"
        else:
            member_name = random.choice(lords_members)
            house = "Lords"
        
        # Generate a random date within the last year
        days_ago = random.randint(0, 365)
        item_date = today - datetime.timedelta(days=days_ago)
        
        # Generate a description that includes the search term
        descriptions = [
            f"The honorable member discussed {term} in relation to economic growth.",
            f"A question was raised about the impact of {term} on public services.",
            f"The debate touched on {term} and its implications for regulation.",
            f"During the discussion on technology policy, {term} was mentioned.",
            f"The member expressed concerns about {term} and data privacy."
        ]
        
        # Create the item
        item = {
            "id": f"sample-{i}",
            "date": item_date.isoformat(),
            "description": random.choice(descriptions),
            "url": "https://hansard.parliament.uk/sample",
            "member": {
                "name": member_name,
                "house": house
            }
        }
        
        items.append(item)
    
    # Sort by date (newest first)
    items.sort(key=lambda x: x["date"], reverse=True)
    
    return {"items": items}

# Process data when user interacts with filters
if selected_terms:
    data = process_hansard_data(selected_terms, selected_house, date_from, date_to)
    
    if data.empty:
        st.info("No data found for the selected filters. Try adjusting your search criteria.")
    else:
        # Display results
        with tab1:
            st.header("Rankings by Mention Frequency")
            
            # Count and display speaker mentions
            speakers_df = count_speaker_mentions(data)
            
            if not speakers_df.empty:
                # Display top 20 speakers
                st.subheader("Top MPs/Lords Mentioning AI-related Terms")
                
                # Apply additional filters if needed
                house_filter = st.selectbox("Filter by House", ["All", "Commons", "Lords"])
                if house_filter != "All":
                    speakers_df = speakers_df[speakers_df['house'] == house_filter]
                
                # Display table with rankings
                st.dataframe(
                    speakers_df.head(20)[['name', 'house', 'count']],
                    column_config={
                        "name": "Name",
                        "house": "House",
                        "count": "Number of Mentions"
                    },
                    use_container_width=True
                )
                
                # Create bar chart of top 10 speakers
                top_speakers = speakers_df.head(10)
                
                chart = alt.Chart(top_speakers).mark_bar().encode(
                    x=alt.X('count:Q', title='Number of Mentions'),
                    y=alt.Y('name:N', title='Name', sort='-x'),
                    color=alt.Color('house:N', title='House'),
                    tooltip=['name:N', 'house:N', 'count:Q']
                ).properties(
                    title='Top 10 MPs/Lords Mentioning AI-related Terms',
                    width=700,
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No speaker information found in the data.")
        
        with tab2:
            st.header("Trends Over Time")
            
            # Create and display trend chart
            trend_chart = create_trend_chart(data)
            if trend_chart:
                st.altair_chart(trend_chart, use_container_width=True)
            else:
                st.info("Insufficient data to create trend chart.")
            
            # Create and display word cloud
            st.subheader("Word Cloud of Discussions")
            wordcloud_fig = create_word_cloud(data)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.info("Insufficient data to create word cloud.")
        
        with tab3:
            st.header("Term Comparison")
            
            # Create and display term frequency chart
            term_chart = create_term_frequency_chart(data)
            if term_chart:
                st.altair_chart(term_chart, use_container_width=True)
            else:
                st.info("Insufficient data to compare terms.")
        
        with tab4:
            st.header("Recent Mentions")
            
            # Display most recent mentions
            if 'date' in data.columns and 'description' in data.columns and 'member' in data.columns:
                st.subheader("Most Recent Mentions")
                
                for _, row in data.head(10).iterrows():
                    with st.expander(f"{row.get('date', 'Unknown date')} - {row.get('member', {}).get('name', 'Unknown speaker')}"):
                        st.markdown(f"**House:** {row.get('member', {}).get('house', 'Unknown')}")
                        st.markdown(f"**Search Term Found:** {row.get('search_term', 'Unknown')}")
                        st.markdown(f"**Description:** {row.get('description', 'No description available')}")
                        if 'url' in row:
                            st.markdown(f"[View original source]({row['url']})")
            else:
                st.info("No detailed mention data available.")

# Footer
st.markdown("---")
st.markdown("Data source: [Hansard API](https://hansard-api.parliament.uk/)")
st.markdown("Created with Streamlit")
