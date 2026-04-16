import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime
import torch
from transformers import pipeline
import time

# Page config
st.set_page_config(
    page_title="YouTube Engagement Benchmarker",
    page_icon="📊",
    layout="centered",
)

# Session state (stage flow)
if "stage" not in st.session_state:
    st.session_state.stage = "input"


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


@st.cache_resource
def load_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )


def get_sentiment(pipe, text):
    if not text.strip():
        return 0.0
    result = pipe(text[:512], truncation=True)[0]
    label, score = result["label"], result["score"]
    pos = score if label == "POSITIVE" else 0.0
    neg = score if label == "NEGATIVE" else 0.0
    return pos - neg


def extract_title_features(title):
    letters = [c for c in title if c.isalpha()]
    total_letters = len(letters)
    uppercase_letters = sum(1 for c in letters if c.isupper())
    caps_ratio = uppercase_letters / total_letters if total_letters > 0 else 0

    return {
        'title_has_question': int('?' in title),
        'title_has_exclamation': int('!' in title),
        'title_has_pipe': int('|' in title),
        'title_caps_ratio': caps_ratio,
        'title_has_number': int(any(c.isdigit() for c in title))
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
    (0.0,   "Very Low",  "#ef4444"),
    (0.01,  "Low",       "#eab308"),
    (0.025, "Average",   "#3b82f6"),
    (0.04,  "High",      "#22c55e"),
    (0.06,  "Very High", "#a855f7")
]


def get_engagement_band(rate):
    for threshold, label, color in reversed(engagement_categories):
        if rate >= threshold:
            return label, color
    return "Very Low", "#ef4444"


# Input page
if st.session_state.stage == "input":
    st.title("YouTube Engagement Benchmarker")
    st.markdown(
        "<p style='color:#888888;margin-top:-0.5rem;'>Compare your video's metadata against North American trending videos.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Video pre-post data")

    title_input = st.text_input("Video title")
    description_input = st.text_area("Description", height=125)
    tags_input = st.text_input("Tags (comma-separated)")

    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Country", ["US", "CA", "MX"])
    with col2:
        category_name = st.selectbox("Category", list(category_map.keys()))

    col3, col4 = st.columns(2)

    if "post_time" not in st.session_state:
        st.session_state.post_time = datetime.now().time()

    with col3:
        post_time = st.time_input("Upload time (UTC)", value=st.session_state.post_time)
    with col4:
        date_to_post = st.date_input("Date to post")

    comments_disabled = st.toggle("Comments disabled", value=False)

    st.divider()

    if st.button("Predict engagement", type="primary"):
        time.sleep(0.6)
        if not title_input.strip():
            st.warning("Please enter a video title.")
            st.stop()
        if not description_input.strip():
            st.warning("Please enter a video description.")
            st.stop()

        # Save inputs
        st.session_state.inputs = {
            "title": title_input,
            "description": description_input,
            "tags": tags_input,
            "country": country,
            "category_name": category_name,
            "post_time": post_time,
            "date": date_to_post,
            "comments_disabled": comments_disabled
        }

        st.session_state.stage = "processing"
        st.rerun()


# Processing page
elif st.session_state.stage == "processing":

    st.markdown("## 🔍 Assessing your video...")

    progress = st.progress(0)
    status = st.empty()

    steps = [
        "Analyzing title sentiment...",
        "Understanding description...",
        "Evaluating pre-post metadata...",
        "Running model prediction...",
        "Finalizing insights..."
    ]

    for i, step in enumerate(steps):
        status.text(step)
        progress.progress((i + 1) / len(steps))
        time.sleep(0.6)

    st.session_state.stage = "results"
    st.rerun()


# Results page
elif st.session_state.stage == "results":
    time.sleep(0.6)
    inputs = st.session_state.inputs

    model, transformer, stats = load_artifacts()
    pipe = load_pipeline()

    title_input = inputs["title"]
    description_input = inputs["description"]
    tags_input = inputs["tags"]
    category_name = inputs["category_name"]

    # Feature engineering
    title_sentiment = get_sentiment(pipe, title_input)
    description_sentiment = get_sentiment(pipe, description_input)

    tag_count = len([t.strip() for t in tags_input.split(",") if t.strip()])
    title_length = len(title_input)
    publish_hour = inputs["post_time"].hour
    publish_day = inputs["date"].weekday()
    category_id = category_map[category_name]
    tf = extract_title_features(title_input)

    new_data = pd.DataFrame([{
        "title_sentiment": title_sentiment,
        "description_sentiment": description_sentiment,
        "tag_count": tag_count,
        "title_length": title_length,
        "category_id": category_id,
        "comments_disabled": inputs["comments_disabled"],
        "publish_hour": publish_hour,
        "publish_day": publish_day,
        "country": inputs["country"],
        **tf,
    }])

    X_new = transformer.transform(new_data)
    log_pred = model.predict(X_new)[0]
    predicted_rate = np.expm1(log_pred)

    percentile_rank = int(np.searchsorted(stats["percentiles"], predicted_rate))
    percentile_rank = max(0, min(100, percentile_rank))

    band_label, band_color = get_engagement_band(predicted_rate)

    # Results
    st.title("Results")
    st.markdown("After 48-72 hours, your video is expected...")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### To have")
        #st.markdown(f"**{band_label}**")
        st.markdown(f"<span style='color:{band_color}'>{band_label}</span>", unsafe_allow_html=True)
        st.markdown("engagement rate")

    with c2:
        st.markdown("### To engage")
        st.markdown(f"**{predicted_rate * 100:.2f}%**")
        st.markdown("of viewers")

    with c3:
        st.markdown("### To beat")
        st.markdown(f"**{percentile_rank}%**")
        st.markdown("of other videos")

    # Graph
    st.divider()
    st.subheader("Performance")

    st.progress(percentile_rank / 100)
    st.caption(f"Your video is expected to be more engaging than {percentile_rank}% of trending videos.")

    chart_data = pd.DataFrame({
        "Metric": ["Your Video", "Average"],
        "Engagement Rate": [predicted_rate, stats["mean"]]
    })

    st.bar_chart(chart_data.set_index("Metric"))

    # Suggestions
    st.divider()
    st.subheader("Improvement Suggestions")

    suggestions = []

    if predicted_rate < stats["mean"]:
        suggestions.append("Your engagement is below average — improve your hook.")

    if title_length < 30:
        suggestions.append("Try a longer title for more context.")

    if tf["title_has_number"] == 0:
        suggestions.append("Try using a number in your title.")

    if tf["title_has_exclamation"] == 0:
        suggestions.append("Add punctuation (! or ?) to increase curiosity.")

    if tag_count < 5:
        suggestions.append("Add more tags to improve discoverability in the algorithm.")

    if not suggestions:
        suggestions.append("Your video is well-optimized. Focus on thumbnail & video content.")

    for s in suggestions:
        st.markdown(f"- {s}")

    # Reset
    st.divider()
    if st.button("Try another video"):
        st.session_state.stage = "input"
        st.rerun()