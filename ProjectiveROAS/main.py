import streamlit as st
import tabs.quarterly_pricing as quarterly_pricing
import tabs.sleep_coach_pricing as sleep_coach_pricing


# --- VARIABLES
tab_names = ["Quarterly Pricing", "Sleep Coach Pricing"]

#
st.title("Models Playground")

# --- Initialize session state
if 'show_sidebar' not in st.session_state:
    st.session_state['show_sidebar'] = True
if 'tab_state' not in st.session_state:
    st.session_state['tab_state'] = tab_names[0]

# Define a function to toggle the sidebar
def toggle_sidebar(show: bool = None):
    st.session_state['show_sidebar'] = show if show else not st.session_state['show_sidebar'] 

# --- Initialize tabs
tabs = st.tabs(tab_names)
with tabs[0]:
    quarterly_pricing.render()
    st.session_state['tab_state'] = tab_names[0]


with tabs[1]:
    sleep_coach_pricing.render()
    st.session_state['tab_state'] = tab_names[1]

# --- init sidebar
if st.session_state['show_sidebar']:
    with st.sidebar:
        if st.session_state['tab_state'] == tab_names[0]:
            quarterly_pricing.render_sidebar()
        elif st.session_state['tab_state'] == tab_names[1]:
            sleep_coach_pricing.render_sidebar()

