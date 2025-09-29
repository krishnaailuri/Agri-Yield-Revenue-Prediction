import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load trained pipeline
# ---------------------------
with open("crop_price_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# ---------------------------
# 2. Page config & header
# ---------------------------
st.set_page_config(
    page_title="Corn Yield & Revenue Prediction",
    layout="wide",
    page_icon="🌽"
)

# Hero section
st.markdown("""
<div style="text-align:center">
    <h1>🌽 Corn Yield & Revenue Prediction</h1>
    <p>Predict corn production per acre and estimate revenue based on key inputs.</p>
    # <img src="https://cdn.pixabay.com/photo/2016/11/19/18/06/corn-1834703_1280.jpg" width="500">
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# 3. Sidebar: Inputs
# ---------------------------
st.sidebar.header("Input Parameters")

year = st.sidebar.number_input("Year", min_value=1900, max_value=2100, value=2025, step=1)
state_ansi = st.sidebar.number_input("State ANSI", min_value=0, max_value=100, value=30, step=1)

period_options = ["YEAR", "YEAR - AUG FORECAST", "YEAR - JUN FORECAST", 
                  "YEAR - NOV FORECAST", "YEAR - OCT FORECAST", "YEAR - SEP FORECAST"]
period = st.sidebar.selectbox("Period", period_options)

state_options = [
    "ALABAMA","ARIZONA","ARKANSAS","CALIFORNIA","COLORADO","CONNECTICUT",
    "DELAWARE","FLORIDA","GEORGIA","IDAHO","ILLINOIS","INDIANA","IOWA",
    "KANSAS","KENTUCKY","LOUISIANA","MAINE","MARYLAND","MASSACHUSETTS",
    "MICHIGAN","MINNESOTA","MISSISSIPPI","MISSOURI","MONTANA","NEBRASKA",
    "NEVADA","NEW HAMPSHIRE","NEW JERSEY","NEW MEXICO","NEW YORK","NORTH CAROLINA",
    "NORTH DAKOTA","OHIO","OKLAHOMA","OREGON","OTHER STATES","PENNSYLVANIA",
    "RHODE ISLAND","SOUTH CAROLINA","SOUTH DAKOTA","TENNESSEE","TEXAS",
    "UTAH","VERMONT","VIRGINIA","WASHINGTON","WEST VIRGINIA","WISCONSIN","WYOMING"
]
state = st.sidebar.selectbox("State", state_options)

data_item_options = ["CORN, GRAIN - ACRES HARVESTED",
                     "CORN, GRAIN - PRODUCTION, MEASURED IN BU",
                     "CORN, SILAGE - ACRES HARVESTED",
                     "CORN, SILAGE - YIELD, MEASURED IN TONS / ACRE"]
data_item = st.sidebar.selectbox("Data Item", data_item_options)

price_per_ton = st.sidebar.number_input("Expected price per ton ($)", min_value=0.0, value=200.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.info("All other parameters (Domain, Category, Commodity) are fixed to TOTAL, NOT SPECIFIED, CORN")

# ---------------------------
# 4. Predict button
# ---------------------------
if st.button("Predict"):
    # Prepare input
    input_df = pd.DataFrame({
        "Year": [year],
        "State ANSI": [state_ansi],
        "Period": [period],
        "State": [state],
        "Data Item": [data_item],
        "Domain": ["TOTAL"],
        "Domain Category": ["NOT SPECIFIED"],
        "Commodity": ["CORN"]
    })
    
    # Prediction
    pred_bu = pipeline.predict(input_df)[0]
    pred_ton = pred_bu * 0.0254
    pred_price_ton = pred_ton * price_per_ton
    pred_price_bu = pred_price_ton / pred_bu
    
    # ---------------------------
    # 5. Display results in colored cards
    # ---------------------------
    st.markdown("### Predicted Production & Revenue")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Production (Bushels) per Acre", value=f"{pred_bu:,.0f} bu/acre")
    col2.metric(label="Production (Tons) per Acre", value=f"{pred_ton:,.2f} t/acre")
    col3.metric(label="Estimated Revenue", value=f"${pred_price_ton:,.2f}")
    
    st.markdown(f"<p style='text-align:center; font-size:18px;'>Price per Bushel: ${pred_price_bu:,.2f}</p>", unsafe_allow_html=True)
    
    # ---------------------------
    # 6. Optional: plot predicted vs typical historical yield
    # ---------------------------
    st.markdown("---")
    st.markdown("### Production Distribution")
    
    # Example: simulate historical distribution for visualization
    hist_yields = np.random.normal(loc=pred_bu, scale=pred_bu*0.15, size=100)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(hist_yields, bins=20, color="cornflowerblue", kde=True, ax=ax)
    ax.axvline(pred_bu, color='orange', linestyle='--', label='Predicted Yield')
    ax.set_xlabel("Yield (Bushels per Acre)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    st.success("🌽 Prediction complete! Use the sidebar to change inputs and predict again.")
