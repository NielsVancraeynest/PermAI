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
voorschrijften_lijst = ["Schrijnberg", "Rozenberg", "De Heide"]
punten_lijst = ["P.4) Tuinhuizen moeten op ten minste 3m van...", "P.2) Tuinhuizen groter dan 40m moeten...","P.17) Tuinhuizen voorzien van elektrische..."]

    
def show_page():
    def show_results():
        def letter_to_int(letter: str):
            alphabet = list('abcdefghijklmnopqrstuvwxyz-_ ')
            return alphabet.index(str.lower(letter)) + 1
        if soort_voorschriften != None  and voorschriften_zoeken != None:
            data_found = 0
            for i in range(len(soort_voorschriften)):
                data_found += letter_to_int(soort_voorschriften[i])
            data_found = int(round(data_found/6,0))
            # data_found = int(round((len(soort_voorschriften))*1.2,0))
            result.write(f"Er zijn {data_found} resultaten gevonden voor {voorschriften_zoeken}.")
            for i in range(data_found):
                index = i%len(voorschrijften_lijst)
                with result.expander(f"**{soort_voorschriften} {voorschrijften_lijst[index]}**"):
                    st.write(punten_lijst[index])

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
        result = st.container(border=False,height=300)

        
                        
                    
                
                