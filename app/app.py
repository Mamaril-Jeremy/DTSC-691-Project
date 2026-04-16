import streamlit as st

pg = st.navigation([
    st.Page("pages/1_biography.py", title="Biography", icon="👤"),         
    st.Page("pages/2_resume.py", title="Resume", icon="📄"),       
    st.Page("pages/3_projects.py", title="Projects", icon="🛠️"),            
    st.Page("pages/4_user_interface.py", title="YouTube Engagement Predictor", icon="📊")   
], position="top")

pg.run()