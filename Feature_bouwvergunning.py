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



lijst_voorschriften = ("RUP", "BPA", "APA")



    
def show_page():
    def show_results():
        if soort_voorschriften != None  and voorschriften_zoeken != None:
            amount = 28
            result.write(f"Er zijn {amount} resultaten gevonden.")
            for i in range(amount):
                with result.expander("**Uitvoeringsbesluit 'Meldingsplicht 16/7/2021**"):
                    st.write("""
                        Art 1.4) Tuinhuizen als vrijstaande volumes
                                """)

    # Download the template based on the value dictionary
    # st.sidebar.download_button("Download template", pd.DataFrame([values.keys()]).to_csv(sep = ';',header = False ,index=False), file_name="template.csv")
    # Element to upload a file
    # uploadedFile  = st.sidebar.file_uploader("Import the excel with all your data",type=["xlsx"])
        
        
    # _______________Main screen_______________
    emptycol1,col,emptycol2 =st.columns([1,3,1])
    
    with col:
        voorschriften_zoeken = st.text_input("Op welk keyword wil je in de bouwvoorschriften zoeken?", placeholder="vb:tuinhuis", key="bouwvergunning")

        soort_voorschriften = st.selectbox("Welk soort voorschrift zoekt u?",
            lijst_voorschriften,
            index=None,
            placeholder="Selecteer een decreet" )
        
        st.button("Zoeken", on_click=lambda: show_results())
        result = st.container(border=False,height=450)

        
                        
                    
                
                
