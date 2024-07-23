import streamlit as st
import tabs.quarterly_pricing as quarterly_pricing
import tabs.sleep_coach_pricing as sleep_coach_pricing


st.title("Models Playground")

tabs = st.tabs(["Quarterly Pricing", "Sleep Coach Pricing"])

with tabs[0]:
    quarterly_pricing.render()

with tabs[1]:
    sleep_coach_pricing.render()


