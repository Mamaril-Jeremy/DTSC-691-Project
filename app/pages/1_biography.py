import streamlit as st
from pathlib import Path

st.title("About Me")
st.subheader("Jeremy Mamaril")

BASE_DIR = Path(__file__).resolve().parent.parent
image_path = BASE_DIR / "static" / "grad_pic.jpg"

st.image(image_path)

st.divider()

st.markdown("Hello! Welcome to my page. I recently graduated with a Bachelor's Degree in Computer Science and Engineering at the University of Nevada, Reno.")
st.markdown("- I started my Master's Degree in Data Science here at Eastern University a bit over a year ago, and I'm excited to graduate again!!")
st.markdown("- Yes, the job market is bad, but I still have a dream of becoming an engineer in the AI or Machine Learning fields.")
st.markdown("- I currently have skills in Python, Tableau, and machine learning. I want to continue learning about machine learning infrastructure and AI prompt engineering.")
st.markdown("- I am mostly interested in using both structured and non-structured data to solve real-world problems.")
st.markdown("- Outside of programming and studying, I enjoy playing basketball, pickleball, and flag football.")
st.markdown("- I also play the piano and video games often.")
