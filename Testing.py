import streamlit as st

st.markdown("""
<style>
label[for*="streamlit"] {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p style="color:white;">Choose the order type</p>', unsafe_allow_html=True)
choice = st.radio("", ["A", "B"])