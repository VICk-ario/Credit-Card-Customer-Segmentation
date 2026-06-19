import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE SETUP & DATA LOADING
# ==========================================
st.set_page_config(page_title="Customer Segmentation App", page_icon="💳", layout="wide")

# Use st.cache_resource so we only load the models once (makes the app fast)
@st.cache_resource
def load_models():
    kmeans = joblib.load('C:\\customer_segmentation\\Credit-Card-Customer-Segmentation\\notebooks\\kmeans_model.pkl')
    scaler = joblib.load('C:\\customer_segmentation\\Credit-Card-Customer-Segmentation\\notebooks\\data_scaler.pkl')
    return kmeans, scaler

@st.cache_data
def load_data():
    return pd.read_csv('C:\\customer_segmentation\\Credit-Card-Customer-Segmentation\\notebooks\\segmented_customers.csv')

kmeans, scaler = load_models()
df = load_data()

# Define our Business Personas based on k=5
# (You may need to adjust the cluster numbers 0-4 based on how your specific model assigned them)
cluster_names = {
    0: "The Sleepers (Inactive)",
    1: "The Whales (High-Net-Worth Loyalists)",
    2: "The Revolvers (Lucrative but Risky)",
    3: "The Transactors (Everyday Users)",
    4: "The Cash Advancers (Emergency Users)"
}

cluster_strategies = {
    0: "Send 'We miss you' activation campaigns with a one-time spending bonus.",
    1: "Offer premium rewards, luxury travel perks, and concierge services.",
    2: "Offer balance transfer incentives to lock them in, but monitor for default risk.",
    3: "Provide cash-back incentives on everyday categories (groceries/gas) to increase volume.",
    4: "Target with lower-interest personal loans to transition them to sustainable debt."
}

# ==========================================
# 2. SIDEBAR: THE "WHAT-IF" SIMULATOR
# ==========================================
st.sidebar.header("Customer Profiler Simulator")
st.sidebar.markdown("Adjust the sliders to simulate a new customer and see which segment they fall into.")

# Create sliders for the most important business features
# Note: In a real app, you would have sliders for all features your model was trained on. 
# For this example, we assume default median values for features not explicitly adjusted here.
def get_user_input():
    user_data = {col: df[col].median() for col in df.columns if col != 'Cluster'}
    
    st.sidebar.markdown("### Financial Metrics")
    user_data['BALANCE'] = st.sidebar.slider("Current Balance ($)", 0.0, 10000.0, 1000.0)
    user_data['PURCHASES'] = st.sidebar.slider("Monthly Purchases ($)", 0.0, 5000.0, 500.0)
    user_data['CASH_ADVANCE'] = st.sidebar.slider("Cash Advances ($)", 0.0, 5000.0, 0.0)
    user_data['CREDIT_LIMIT'] = st.sidebar.slider("Credit Limit ($)", 500.0, 20000.0, 3000.0)
    user_data['PAYMENTS'] = st.sidebar.slider("Payments Made ($)", 0.0, 5000.0, 800.0)
    
    st.sidebar.markdown("### Behavioral Metrics")
    # New Slider 1: How often do they use the card?
    user_data['PURCHASES_FREQUENCY'] = st.sidebar.slider(
        "Purchase Frequency (0 = Never, 1 = Every Day)", 
        0.0, 1.0, 0.5
    )
    
    # New Slider 2: Do they carry debt?
    user_data['PRC_FULL_PAYMENT'] = st.sidebar.slider(
        "Full Payment % (0 = Minimums, 1 = Pays in Full)", 
        0.0, 1.0, 0.0
    )
    
    return pd.DataFrame([user_data])

user_input_df = get_user_input()

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
# Scale the user's input using the exact same scaler from Notebook 2
user_input_scaled = scaler.transform(user_input_df)

# Predict the cluster
predicted_cluster = kmeans.predict(user_input_scaled)[0]
predicted_persona = cluster_names[predicted_cluster]
recommended_strategy = cluster_strategies[predicted_cluster]

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.title("💳 Credit Card Customer Segmentation Engine")
st.markdown("This tool uses K-Means clustering to segment credit card users into 5 distinct business personas.")

# Create tabs for clean organization
tab1, tab2 = st.tabs(["🎯 Live Prediction", "📊 Cluster Analysis"])

with tab1:
    st.subheader("Simulated Customer Profile")
    st.write("Based on the sidebar parameters, this new customer is classified as:")
    
    # Use columns to make it look like a dashboard metric card
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Customer Persona:** \n### {predicted_persona}")
    with col2:
        st.success(f"**Marketing Strategy:** \n{recommended_strategy}")

with tab2:
    st.subheader("Interactive Cluster Profiling")
    st.markdown("How do the 5 segments compare across key financial metrics?")
    
    # Calculate means for the radar chart
    cluster_means = df.groupby('Cluster').mean().reset_index()
    features_to_plot = [
    'BALANCE', 'PURCHASES', 'CASH_ADVANCE', 
    'CREDIT_LIMIT', 'PAYMENTS', 
    'PURCHASES_FREQUENCY', 'PRC_FULL_PAYMENT'
                 ]
    
    # Build an interactive Plotly Radar Chart
    fig = go.Figure()
    for i in range(5):
        fig.add_trace(go.Scatterpolar(
            r=cluster_means.loc[i, features_to_plot].values,
            theta=features_to_plot,
            fill='toself',
            name=cluster_names[i]
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Average Financial Behaviors by Segment"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add an interactive scatter plot (PCA representation if desired, or simple 2D)
    st.markdown("### Customer Distribution")
    st.markdown("Explore how individual customers map against two variables.")
    x_axis = st.selectbox("X-Axis", features_to_plot, index=3) # Default Credit Limit
    y_axis = st.selectbox("Y-Axis", features_to_plot, index=0) # Default Balance
    
    scatter_fig = px.scatter(
        df, x=x_axis, y=y_axis, 
        color=df['Cluster'].map(cluster_names),
        opacity=0.6,
        title=f"Customer Segmentation: {x_axis} vs {y_axis}"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)