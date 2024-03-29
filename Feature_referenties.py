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



lijst_gemeentes = ("x gemeente", "y_gemeente", "z_gemeente")
lijst_straatnamen = ["Bremstraat", "Hofstraat", "werfweg"]


    
def show_page():
    def letter_to_int(letter: str):
        alphabet = list('abcdefghijklmnopqrstuvwxyz-_ ')
        return alphabet.index(str.lower(letter)) + 1
    def show_results():
        if referentie_gemeente !=None and referentie_zoeken != "":
                data_found = int(round((len(referentie_zoeken) + len(referentie_gemeente))*2.5,0))
                result.write(f"Er zijn {data_found} resultaten gevonden.")
                startDate = datetime(2024, 3, 28) - timedelta(days=data_found)
                for i in range(data_found):
                    with result.container(border=True):
                        col1,col2 =st.columns([3,1])
                        with col1:
                            generated_index = letter_to_int((3*(referentie_zoeken + referentie_gemeente))[i])
                            index_random_straat = generated_index%len(lijst_straatnamen)
                            random_huisnr = generated_index*int(round((len(referentie_zoeken)+len(referentie_gemeente))/1.5,0))
                            st.markdown(f"**Vergunning {referentie_zoeken} {lijst_straatnamen[index_random_straat]} {random_huisnr} te {referentie_gemeente}**")
                            st.download_button("Download pdf", 'test.xlsx', file_name=f"Vergunning {referentie_zoeken} {lijst_straatnamen[index_random_straat]} {random_huisnr} te {referentie_gemeente}.pdf", key=i,
                                               on_click=lambda: show_results())
                        with col2:
                            st.markdown(f"\n*Verleend op {startDate.day}/{startDate.month}/{startDate.year}*")
                            startDate = startDate - timedelta(days=random_huisnr/3)

    # Download the template based on the value dictionary
    # st.sidebar.download_button("Download template", pd.DataFrame([values.keys()]).to_csv(sep = ';',header = False ,index=False), file_name="template.csv")
    # Element to upload a file
    # uploadedFile  = st.sidebar.file_uploader("Import the excel with all your data",type=["xlsx"])
        
        
    # _______________Main screen_______________
    emptycol1,col,emptycol2 =st.columns([1,3,1])
    
    with col:
        referentie_zoeken = st.text_input("Op welk keyword wil je een vergunning zoeken?", placeholder="vb:tuinhuis")

        referentie_gemeente = st.selectbox(label="In welke gemeente wilt u zoeken?",options=lijst_gemeentes,index=0,*"Selecteer een gemeente" )
        
        st.button("Zoeken", on_click=lambda: show_results())
        result = st.container(border=False,height=300)

        
                        
                    
                
                
