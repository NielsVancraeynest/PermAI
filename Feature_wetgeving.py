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
wetten_lijst = ["'Meldingsplicht' 16/7/2021", "'Vrijgestelde handelingen'", "'Afwijkingen'"]
artikelen_lijst = ["Art 1.4) Tuinhuizen als vrijstaande volumes.", "Art 1.1.2°A) Tuinhuizen < 40m^2 hoeven geen vergunning te hebben.",
                   "Art 7.2) Tuinhuizen die voorzien zijn van..."]

    
def show_page():
    def show_results():
        if soort_discreet != None  and wetgeving_zoeken != None:
            data_found = int(round((len(soort_discreet) + len(wetgeving_zoeken))*0.8,0))
            result.write(f"Er zijn {data_found} resultaten gevonden voor {wetgeving_zoeken}.")
            for i in range(data_found):
                index = i%len(wetten_lijst)
                with result.expander(f"**{soort_discreet} {wetten_lijst[index]}**"):
                    st.write(artikelen_lijst[index])

    # Download the template based on the value dictionary
    # st.sidebar.download_button("Download template", pd.DataFrame([values.keys()]).to_csv(sep = ';',header = False ,index=False), file_name="template.csv")
    # Element to upload a file
    # uploadedFile  = st.sidebar.file_uploader("Import the excel with all your data",type=["xlsx"])
        
        
    # _______________Main screen_______________
    emptycol1,col,emptycol2 =st.columns([1,3,1])
    
    with col:
        wetgeving_zoeken = st.text_input("Op welk keyword wil je in de wetgeving zoeken?", placeholder="vb:tuinhuis", key="wetgeving")

        soort_discreet = st.selectbox(label ="Welk soort Decreet zoekt u?",
            options=lijst_decreet,
            index=None,
            placeholder="Selecteer een decreet" )
        
        st.button("Zoeken", on_click=lambda: show_results())
        result = st.container(border=False,height=300)

        
                        
                    
                
                