from ast import For
from cmath import sqrt, tan
import imp
from math import pi
import text


from ssl import ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageOps
from st_clickable_images import clickable_images
from datetime import datetime, timedelta



lijst_decreet = ("VCRO", "Decreet omgevingsvergunningen", "Uitvoeringsbesluiten")



    
def show_page():
    
    # Download the template based on the value dictionary
    # st.sidebar.download_button("Download template", pd.DataFrame([values.keys()]).to_csv(sep = ';',header = False ,index=False), file_name="template.csv")
    # Element to upload a file
    # uploadedFile  = st.sidebar.file_uploader("Import the excel with all your data",type=["xlsx"])
        
        
    # _______________Main screen_______________
    emptycol1,col,emptycol2 =st.columns([1,3,1])
    
    with col:
        wetgeving_zoeken = st.text_input("Op welk keyword wil je in de wetgeving zoeken?", placeholder="vb:tuinhuis", key="wetgeving")

        soort_discreet = st.selectbox("Welk soort Decreet zoekt u?",
            lijst_decreet,
            index=0)
        
        st.button("Zoeken", on_click=lambda: show_results())
        result = st.container()

    def show_results():
        if soort_discreet != None  and wetgeving_zoeken != None:
            amount = 28
            result.write(f"Er zijn {amount} resultaten gevonden.")
            for i in range(amount):
                with result.expander("**Uitvoeringsbesluit 'Meldingsplicht 16/7/2021**"):
                    st.write("""
                        Art 1.4) Tuinhuizen als vrijstaande volumes
                                """)

                        
                    
                
                
