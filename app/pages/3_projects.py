import streamlit as st

st.title("General Projects")
st.subheader("Plato")
st.markdown("Plato is a web-based application designed to cultivate meaningful mentor-mentee relationships within a community of professional users. Mentors and mentees provide essential information by creating custom profiles which are used to enhance matchmaking.")
st.link_button(url="https://www.youtube.com/watch?v=cVx3tBaflXY", label="YouTube Video")
st.link_button(url="https://github.com/Mamaril-Jeremy/CS425-Project", label="GitHub Repo")

st.text("")

st.subheader("Fake Review Detector")
st.markdown("Created a project that goes through AI process to build and compare multiple machine learning models to detect whether a text review is truthful or deceptive (fake).")
st.link_button(url="https://github.com/McIlwee-Nevan/CS482-Final-Project_BeierMamarilMcIlwee/tree/main", label="Fake Review Detector")

st.text("")

st.subheader("Swamp Cooler Project")
st.markdown("Created a swamp cooler that transitions through several states depending on whether or not the temperature of the water or the water level is below or above a threshold.")
st.markdown("The states that the cooler is running in are running, idle, error, and disabled, and the current state it is running in is indicated by lighting the corresponding LED. Output is also displayed by the serial monitor and the LCD display.")
st.markdown("Utilized the Arduino Mega 2560 and the corresponding software.")
st.link_button(url="https://github.com/Mamaril-Jeremy/CPE-301-Final-Project", label="Swamp Cooler")
