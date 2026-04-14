import streamlit as st

st.title("Resume Page")

st.subheader("Education")
st.markdown("**University of Nevada, Reno**")
st.markdown("Aug 2021 to Dec 2024")
st.markdown("- Graduated with a B.S. in Computer Science and Engineering and with a minor in Mathematics")
st.markdown("- Dean’s list all semesters")
st.markdown("- Member of the Honors College (1874 Scholar)")

st.text("")

st.subheader("Work Experience")
st.markdown("**Mathematics Tutor at the University of Nevada, Reno**")
st.markdown("Nov 2021 – Jan 2024 | Aug 2024 – Dec 2024")
st.markdown("- Delivered personalized tutoring sessions to over 1,000 students from diverse academic backgrounds")
st.markdown("- Utilized various teaching methods, such as visual aids, real-world examples, and problem-solving techniques, to ensure students’ mastery of mathematical principles")
st.markdown("- Conducted exam review sessions to improve students' comprehension and to address questions effectively")

st.text("")

st.markdown("**Teaching Assistant at the University of Nevada, Reno**")
st.markdown("Jan 2024 – May 2024")
st.markdown("- Developed supplementary materials, such as example problems and code samples, in the Analysis of Algorithms course")
st.markdown("- Provided personalized feedback and guidance on homework assignments")
st.markdown("- Conducted comprehensive review sessions by discussing complex algorithmic concepts and problem-solving strategies, significantly improving students' understanding")

st.text("")

st.subheader("Clubs and Volunteering Experience")
st.markdown("**NevadaFIT Mentor at the University of Nevada, Reno**")
st.markdown("Aug 2022 | Aug 2023")
st.markdown("- Led study groups, academic skills workshops, and Pack Mentor sessions for incoming freshmen")
st.markdown("- Trained mentees on how to cultivate resilience and perseverance")

st.markdown("**Every Nation Campus Club at the University of Nevada, Reno**")
st.markdown("Feb 2024 - Dec 2024")
st.markdown("- Facilitated Bible studies in community groups to inspire meaningful discussions and spiritual growth")
st.markdown("- Coordinated church operations for well-organized events and services")

st.subheader("View Some of My Projects Here")
if st.button("Project Page"):
    st.switch_page("pages/3_projects.py")