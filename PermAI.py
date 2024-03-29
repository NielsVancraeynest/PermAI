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
from time import sleep
from hydralit_components import HyLoader, Loaders
import PermAI




    
def show_page():
    import streamlit as st
    import streamlit.components.v1 as components
    import pandas as pd
    import base64
    import json
    def download_button(object_to_download, download_filename):
        """
        Generates a link to download the given object_to_download.
        Params:
        ------
        object_to_download:  The object to be downloaded.
        download_filename (str): filename and extension of file. e.g. mydata.csv,
        Returns:
        -------
        (str): the anchor tag to download object_to_download
        """
        if isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(object_to_download.encode()).decode()

        except AttributeError as e:
            b64 = base64.b64encode(object_to_download).decode()

        dl_link = f"""
        <html>
        <head>
        <title>Start Auto Download file</title>
        <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
        <script>
        $('<a href="data:text/csv;base64,{b64}" download="{download_filename}">')[0].click()
        </script>
        </head>
        </html>
        """
        return dl_link

    def show_results():
        if file != None:
            with result.status("Perm**A:blue[I] Smarter** *permits*, :blue[**faster** *approvals*]", expanded=True) as status:
                st.write("Document scannen...")
                sleep(2)
                st.write("Voorbereiden om features te doorlopen...")
                sleep(1)
                st.write(":blue[- referenties]")
                sleep(3)
                st.write(":blue[- wetgeving]")
                sleep(2)
                st.write(":blue[- bouwvoorschriften]")
                sleep(3)
                st.write("Finaal document genereren...")
                sleep(2)
                components.html(
                    download_button(pd.DataFrame(), "PermAI_" + file.name +".pdf"))
                    

    # _______________Main screen_______________
    emptycol1,col,emptycol2 =st.columns([1,3,1])
    
    with col:
        file = st.file_uploader("Importeer de aanvraag",type=["docx", "pdf"])
        result = st.container(border=False)
        if result.button("Genereer voorstel", on_click=show_results):
            st.rerun()
               

        
                        
                    
                
                