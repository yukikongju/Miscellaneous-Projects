import streamlit as st

description = """
We would like to determine the comission that should be charged for a B2B 
partnership in order to make money (or at least be even) based on a **monthly**
revenue projection. We consider that when integrating B2B into our app,
we should account for the potential paid rate decrease and refund rate increase. 

Subscription Model:
- Monthly Subscription give you access to one session
- User pay for extra session
"""

latex_formula = r"""
\begin{equation}
N = T * r_v * r_c
\end{equation}

\\

\begin{equation}
P_g = p_b + p_{\Delta}
\end{equation}

\\

\begin{equation}
R_g = r_b + r_{\Delta}
\end{equation}

\\

\begin{equation}
R_{BS} = T * (P_g - R_g)
\end{equation}

\\

\begin{equation}
R_{B2B} = N * C + N * S * CA
\end{equation}

\\

\begin{equation}
R = R_{BS} + R_{B2B}
\end{equation}

"""

variables_description = """
- T: Monthly Traffic
- N: Number of Conversions
- rv: % of users who view banners/PAC
- rc: % of users who convert to B2B
- Pg: overall paid rate
- pb: paid rate baseline
- p delta: change in paid rate
- Rg: overall refund rate
- rb: refund baseline
- r delta: change in refund rate
- C: commission per subscription
- CA: commision per additional session
- S: Number of additional session per user per month
- R_{BS}: Revenue from BetterSleep App
- R_{B2B}: Revenue from B2B
- R: Total Revenue
"""

def render():
    # --- Description
    st.write("### How the model work")
    st.write(description)
    st.latex(latex_formula)
    st.write(variables_description)


    # --- Sliders for the parameters
    st.write("### Playground")
    c7, c8, c9, c10 = st.columns(4)
    with c7: 
        T = st.slider("T: Monthly Traffic", 0, 3500000, 1)
    with c8: 
        C = st.slider("C: Comission per conversion", 0.0, 1000.0, 0.5)
    with c9:
        S = st.slider("S: Number of additional sessions per user", 1.0, 30.0, 0.01)
    with c10:
        CA = st.slider("CA: Commision per additional session ", 0.0, 1000.0, 0.5)

    c1, c2 = st.columns(2)
    with c1: 
        r_v = st.slider("r_v: banner/PAC view rate", 0.0, 100.0, 0.1) / 100
    with c2:
        r_c = st.slider("r_c: B2B conversion rate", 0.0, 100.0, 0.01) / 100

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        p_b = st.slider("p_b: paid rate baseline", 0.0, 100.0, 0.01) / 100
    with c4:
        p_delta = st.slider("p_delta: paid rate diff", 0.0, 100.0, 0.01) / 100
    with c5:
        r_b = st.slider("r_b: refund rate baseline", 0.0, 100.0, 0.01) / 100
    with c6:
        r_delta = st.slider("r_delta: refund rate diff", 0.0, 100.0, 0.01) / 100


    N_val = T * r_v * r_c
    P_g = p_b + p_delta
    R_g = r_b + r_delta
    R_BS = T * (P_g - R_g)
    R_B2B = N_val * C + N_val * S * CA
    R = R_BS + R_B2B

    # --- 
    st.write(f"**Estimated number of conversions:** {N_val}")
    st.write(f"**Revenue BetterSleep:** {R_BS:.2f}$")
    st.write(f"**Revenue B2B:** {R_B2B:.2f}$")
    st.write(f"**Total Revenue:** {R:.2f}$")

    # --- 

def render_sidebar():
    st.write("### Model Rates")
    st.selectbox("Model", ["Sleep Coach Pricing"])




