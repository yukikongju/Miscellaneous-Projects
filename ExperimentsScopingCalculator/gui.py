import streamlit as st
from brief import BriefInformation

st.title("Self-served Brief Calculator")

# --- initialize sidebar with Reach and baseline KPI links


def init_sidebar():
    # --- Monthly Reach
    st.sidebar.title("Monthly Reach")

    st.sidebar.markdown("[Home Counts - EN](https://mixpanel.com/s/2ttmGx)")

    # - Reach Growth
    st.sidebar.write("**Growth**")

    # Reach Content
    st.sidebar.write("**Content**")

    st.sidebar.markdown("[Screen Sounds - EN](https://mixpanel.com/s/4COAnQ)")
    st.sidebar.markdown("[Screen Mixes - EN]")
    st.sidebar.markdown("[Feed Meditation - EN]")
    st.sidebar.markdown("[Feed Bedtime - EN]")
    st.sidebar.markdown("[Feed Music - EN]")
    st.sidebar.markdown("[Screen Content Playing](https://mixpanel.com/s/2hpf4H)")
    st.sidebar.markdown("[Screen Mixer](https://mixpanel.com/s/2Rp3Pw)")
    st.sidebar.markdown("[Screen Mixer/Player](https://mixpanel.com/s/3WwpRP)")

    # - Reach Product
    st.sidebar.write("**Product**")
    st.sidebar.markdown("[Recorder Welcome - EN](https://mixpanel.com/s/1v0B5L)")
    st.sidebar.markdown("[Screen Recorder - EN](https://mixpanel.com/s/3zU3E9)")
    st.sidebar.markdown("[Screen Journal - EN](https://mixpanel.com/s/4ADoXV)")
    st.sidebar.markdown("[Screen Journal - Empty State](https://mixpanel.com/s/3xvXyp)")
    st.sidebar.markdown("[Tracker Loaded 1 day - EN](https://mixpanel.com/s/edcAw)")
    st.sidebar.markdown("[Screen Mixer - EN](https://mixpanel.com/s/KJaZC)")

    # --- Baseline
    st.sidebar.title("Baseline")

    st.sidebar.markdown("[Home to Paid - EN](https://mixpanel.com/s/2yOuaf)")
    st.sidebar.markdown("[Home to Subscription 24h in trial - EN](https://mixpanel.com/s/JTVHg)")

    # - baseline growth
    st.sidebar.write("**Growth**")

    # - baseline content
    st.sidebar.write("**Content**")
    st.sidebar.markdown("[Home to Listening - EN](https://mixpanel.com/s/1wXDu0)")
    st.sidebar.markdown("[Screen Sounds to Sound Listening - EN](https://mixpanel.com/s/eAT1h)")
    st.sidebar.markdown("[Screen Sounds to Paid - EN](https://mixpanel.com/s/1X4HF7)")
    st.sidebar.markdown("[Screen Mixes to Mixes Listening - EN]")
    st.sidebar.markdown("[Screen Mixes to Paid - EN]")
    st.sidebar.markdown("[Feed Meditation to Med Listening - EN]")
    st.sidebar.markdown("[Feed Meditation to Paid- EN]")
    st.sidebar.markdown("[Feed Bedtime to Bedtime Listening - EN]")
    st.sidebar.markdown("[Feed Bedtime to Paid - EN]")
    st.sidebar.markdown("[Feed Music to Music Listening - EN]")
    st.sidebar.markdown("[Feed Music to Paid - EN]")
    st.sidebar.markdown("[Home to Mix Listening - EN](https://mixpanel.com/s/3UxYdN)")

    st.sidebar.markdown("[Screen Content Playing to Paid](https://mixpanel.com/s/17lJgs)")
    st.sidebar.markdown(
        "[Screen Content Playing to Guided Listening](https://mixpanel.com/s/2u6DK8)"
    )
    st.sidebar.markdown("[Screen Mixer to Paid](https://mixpanel.com/s/1cMRxV)")
    st.sidebar.markdown("[Screen Mixer to Guided Listening](https://mixpanel.com/s/2rtO2c)")
    st.sidebar.markdown("[Screen Mixer/Player to Paid]")
    st.sidebar.markdown("[Screen Mixer/Player to Guided Listening]")

    # - baseline product
    st.sidebar.write("**Product**")
    st.sidebar.markdown("[Recorder Welcome to Paid - EN](https://mixpanel.com/s/2cnYPd)")
    st.sidebar.markdown(
        "[Recorder Welcome to Tracker Loaded 3 days - EN](https://mixpanel.com/s/85oCZ)"
    )
    st.sidebar.markdown("[Screen Recorder to Paid - EN](https://mixpanel.com/s/4aVBOh)")
    st.sidebar.markdown(
        "[Screen Recorder to Tracker Loaded 3 days - EN](https://mixpanel.com/s/25gXFg)"
    )
    st.sidebar.markdown("[Screen Journal to Paid - EN](https://mixpanel.com/s/2KFQMa)")
    st.sidebar.markdown(
        "[Screen Journal to Tracker Loaded 3 days - EN](https://mixpanel.com/s/43TztA)"
    )
    st.sidebar.markdown("[Screen Journal to Screen Journal - EN](https://mixpanel.com/s/h2inw)")
    st.sidebar.markdown("[Tracker Loaded 1 day to Paid - EN](https://mixpanel.com/s/4G8KIK)")
    st.sidebar.markdown(
        "[Tracker Loaded 1 day to Subsciption 24h in-trial - EN](https://mixpanel.com/s/1ajCrT)"
    )
    st.sidebar.markdown(
        "[Tracker Loaded 1 day to Tracker Loaded 3 days - EN](https://mixpanel.com/s/3D84kv)"
    )
    st.sidebar.markdown("[Sceen Mixer to Tracker Loaded 3 days - EN](https://mixpanel.com/s/XpeUj)")
    st.sidebar.markdown("[Tracker Adoption - EN](https://mixpanel.com/s/122zo0)")


# ------------------------------------------------------------------

def show_brief_results():
    # initialize brief
    brief = BriefInformation(
        monthly_reach=int(reach_input),
        baseline=float(baseline_input),
        experiment_sizing_timeline=int(expected_timeline_input),
        num_variants=num_variants_selectbox + 1,
        alpha=alpha_input,
        power=power_input,
    )

    # compute information
    sample_size_per_bucket, absolute_delta, relative_delta = (
        brief._get_initial_bucket_size_and_deltas()
    )

    # show information
    st.write(f"**Sample Size per bucket**: {round(sample_size_per_bucket)}")
    st.write(f"**Absolute Delta**: {round(absolute_delta, 6)}")
    st.write(f"**Relative Delta**: {round(relative_delta * 100, 3)}%")

    # show additional timeline
    df_timeline = brief._get_additional_timelines()
    st.dataframe(df_timeline)


# ------------------------------------------------------------------

# ---- init mixpanel sidebar links
init_sidebar()

# ---- create input fields for: (1) reach, (2) baseline, (3) week needed
baseline_input = st.text_input("Baseline")
reach_input = st.text_input("Monthly Reach")
expected_timeline_input = st.selectbox(
    label="Expected Timeline (in weeks)", options=[2, 3, 4, 5, 6]
)
num_variants_selectbox = st.selectbox(label="Number of Variants", options=[1, 2, 3, 4])
alpha_input = st.number_input("Alpha", min_value=0.05, max_value=0.15, value=0.1)
power_input = st.number_input("Power", min_value=0.60, max_value=1.0, value=0.8)

# ---- compute brief information if button is pressed
if baseline_input and reach_input and expected_timeline_input and num_variants_selectbox and alpha_input and power_input:
    compute_button = st.button("Compute Timeline")
    if compute_button:
        show_brief_results()

