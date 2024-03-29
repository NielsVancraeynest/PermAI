import streamlit as st
import os

result = os.popen('pip list').read()
print(result)
st.code(result, language=None)