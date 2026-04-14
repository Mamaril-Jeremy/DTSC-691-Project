import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime
import torch
from transformers import pipeline


# Page config
st.set_page_config(
    page_title="YouTube Engagement Benchmarker",
    page_icon="📈",
    layout="centered",
)


# Load artifacts 
@st.cache_resource(show_spinner=False) 
def load_artifacts():
    with open("../models/youtube_model.pkl", "rb") as f:
        model = pkl.load(f)
    with open("../models/column_transformer.pkl", "rb") as f:
        transformer = pkl.load(f)
    with open("../models/engagement_stats.pkl", "rb") as f:
        stats = pkl.load(f)
    return model, transformer, stats

def get_sentiment(pipe, text):
    if not text.strip():
        return 0.0
    result = pipe(text[:512], truncation=True)[0]
    label, score = result["label"], result["score"]
    pos = score if label == "POSITIVE" else 0.0
    neg = score if label == "NEGATIVE" else 0.0
    return pos - neg

@st.cache_resource
def load_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )


def extract_title_features(title):
    letters = [c for c in title if c.isalpha()]

    total_letters = len(letters)
    uppercase_letters = sum(1 for c in letters if c.isupper())

    caps_ratio = uppercase_letters / total_letters if total_letters > 0 else 0

    return {
        'title_has_question'   : int('?' in title),
        'title_has_exclamation': int('!' in title),
        'title_has_pipe'       : int('|' in title),
        'title_caps_ratio'     : caps_ratio
    }


category_map = {
    "Film & Animation": 1,
    "Autos & Vehicles": 2,
    "Music": 10,
    "Pets & Animals": 15,
    "Sports": 17,
    "Short Movies": 18,
    "Travel & Events": 19,
    "Gaming": 20,
    "Videoblogging": 21,
    "People & Blogs": 22,
    "Comedy": 23,
    "Entertainment": 24,
    "News & Politics": 25,
    "Howto & Style": 26,
    "Education": 27,
    "Science & Technology": 28,
    "Nonprofits & Activism": 29,
    "Movies": 30,
    "Anime/Animation": 31,
    "Action/Adventure": 32,
    "Classics": 33,
    "Documentary": 35,
    "Drama": 36,
    "Family": 37,
    "Foreign": 38,
    "Horror": 39,
    "Sci-Fi/Fantasy": 40,
    "Thriller": 41,
    "Shorts": 42,
    "Shows": 43,
    "Trailers": 44,
}


engagement_categories = [
    (0.06,  "Very High", "#a855f7", "Top-tier — viral or highly engaged audience"),
    (0.04,  "High",      "#22c55e", "Strong — above most trending channels"),
    (0.025, "Average",   "#3b82f6", "Normal, healthy performance"),
    (0.01,  "Low",       "#eab308", "Below average — room for improvement"),
    (0.0,   "Very Low",  "#ef4444", "Audience is barely interacting"),
]
 
def get_engagement_band(rate):
    for threshold, label, color, desc in engagement_categories:
        if rate >= threshold:
            return label, color, desc
    return engagement_categories[-1][1], engagement_categories[-1][2], engagement_categories[-1][3]
 

# Header 
st.title("YouTube Engagement Benchmarker")
st.markdown(
    "<p style='color:#888888;margin-top:-0.5rem;'>Compare your video's metadata against North American trending videos.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Input form 
st.subheader("Video pre-post data")

title_input = st.text_input("Video title", placeholder="e.g. I Tried Staying at Stores After Closing")
description_input = st.text_area(
    "Description",
    placeholder="Paste your video description here…",
    height=125,
)
tags_input = st.text_input(
    "Tags (keywords)",
    placeholder="comma-separated, e.g. food, challenge, comics",
)

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", ["US", "CA", "MX"])
active_map = country_maps[country]
with col2:
    category_name = st.selectbox("Category", list(active_map.keys()))
    

col3, col4 = st.columns(2)
if "post_time" not in st.session_state:
    st.session_state.post_time = datetime.now().time()

with col3:
    post_time = st.time_input(
        "Upload time (UTC)",
        value=st.session_state.post_time,
        key="post_time",
    )
with col4:
    date_to_post = st.date_input("Date to post")
 
comments_disabled = st.toggle("Comments disabled", value=False)

st.divider()
run = st.button("Predict engagement", type="primary")

# Prediction 
if run:
    if not title_input.strip():
        st.warning("Please enter a video title.")
        st.stop()

    model, transformer, stats = load_artifacts()

    with st.spinner("Analysing sentiment…"):
        pipe = load_pipeline()
        title_sentiment       = get_sentiment(pipe, title_input)
        description_sentiment = get_sentiment(pipe, description_input)

    tag_count    = len([t.strip() for t in tags_input.split(",") if t.strip()])
    title_length = len(title_input)
    publish_hour = post_time.hour
    publish_day  = date_to_post.weekday()
    category_id = category_map[category_name]
    tf           = extract_title_features(title_input)

    new_data = pd.DataFrame([{
        "title_sentiment"      : title_sentiment,
        "description_sentiment": description_sentiment,
        "tag_count"            : tag_count,
        "title_length"         : title_length,
        "category_id"          : category_id,
        "comments_disabled"    : comments_disabled,
        "publish_hour"         : publish_hour,
        "publish_day"          : publish_day,
        "country"              : country,
        **tf,
    }])

    with st.spinner("Generating prediction…"):
        X_new          = transformer.transform(new_data)
        log_pred       = model.predict(X_new)[0]
        predicted_rate = np.expm1(log_pred)

    # Normalise
    percentile_rank = int(np.searchsorted(stats["percentiles"], predicted_rate))
    percentile_rank = max(0, min(100, percentile_rank))
    p1, p99         = stats["p1"], stats["p99"]
    minmax = (predicted_rate - p1) / (p99 - p1) * 100
    minmax = max(0, min(100, minmax))

    band_label, band_color, band_desc = get_engagement_band(predicted_rate)

    # Results 
    st.divider()
    st.subheader("Results")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Predicted rate",
            f"{predicted_rate * 100:.2f}%",
            help="(likes + comments) / views × 100",
        )
    with m2:
        st.metric(
            "Dataset average",
            f"{stats['mean'] * 100:.2f}%",
            help="Mean across ~48k trending videos",
        )
    with m3:
        st.metric(
            "Percentile",
            f"{percentile_rank}th",
            help="Beats this % of trending videos",
        )
 
    st.divider()
    st.caption(
        "Benchmark is ~48,000 North American trending videos (US/CA/MX). "
    )