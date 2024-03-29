import streamlit as st
import hydralit_components as hc

import Feature_wetgeving
import Feature_referenties
import Feature_bouwvergunning
import PermAI




def show_page():
    # Navigationbar
    menu_data = [
    {'label': 'PermAI'},
    {'label':"Features", 'submenu':[{'icon': "fa fa-paperclip",'label':"Referenties"},
    {'icon': "fa fa-book",'label':"Wetgeving"},
    {'icon': "fa fa-building",'label':"Bouwvoorschriften"}]},
    # {'label':"Referenties"},
    # {'label':"Wetgeving"},
    # {'label':"Bouwvoorschriften"}
    ]

    over_theme = {'txc_inactive': "#82c7a5",'menu_background':'#262730','txc_active': '#0145AC', 'option_active': "#82c7a5"}
    menu_id = hc.nav_bar(
    menu_definition=menu_data,
    first_select= int(0),
    override_theme=over_theme,
    use_animation=False,
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=False, #at the top or not
    sticky_mode='Pinned', #jumpy or not-jumpy, but sticky or pinned
    )


    
    # start a file based on the selected tab in the navigationbar
    if menu_id == "PermAI":
        PermAI.show_page()
    if menu_id == "Referenties":
        Feature_referenties.show_page()
    if menu_id == "Wetgeving":
        Feature_wetgeving.show_page()
    if menu_id == "Bouwvoorschriften":
        Feature_bouwvergunning.show_page()

    
    return 1