import streamlit as st

description = """
Some web campaigns pricing are not yearly, but quarterly. As a result, 
if we only consider the revenues coming from the initial purchase, we would 
be underestimating that campaign ROAS. As such, we need to include the 
revenues we would get from subsequent renewals.

For a quarterly pricing campaigns, revenues would be computed as such:
"""

latex_formula = r"""
\begin{equation}
R = c + \sum_{i=1}^{n} {r_i * c}
\end{equation}
"""

variables_description = """
- R: Yearly Revenue
- c: Cost per period
- r: Renewal rate at the ith period
"""

#  $$\text{Yearly Revenues} = \text{monthly fees} + \sum{ \text{renewal rate} * \text{monthly fees} } $$

def render():
    # --- Description
    st.write("### How the model work")
    st.write(description)
    st.latex(latex_formula)
    st.write(variables_description)

    # ---
    st.write("### Playground")
    period_dct = {
            "Yearly": 1,
            "Quarterly": 4,
            "Monthly": 12,
            }
    period_selectbox = st.selectbox("Pricing Option", period_dct.keys())
    n = period_dct[period_selectbox]
    period_cost = st.slider(f"{period_selectbox} Cost", 0, 100, 25)
    renewal_rates = [st.slider(f"{i}-th renewal rate (%)", 0, 100, 20) for i in range(1, n)]

    revenues = period_cost + sum([r * period_cost / 100 for r in renewal_rates ])
    st.write(f"**Yearly Revenues:** {revenues}$")


def render_sidebar():
    st.write("### Model Rates")
    st.selectbox("Model", ["Quarterly Pricing Model"])


