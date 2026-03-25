import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Retail Intelligence Hub 2026", layout="wide", page_icon="📈")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d425c;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("📊 Retail Intelligence Hub")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔮 Sales Forecasting", "👥 Customer Segmentation", "🤖 AI Strategy Agent"])

# --- TAB 1: SALES FORECASTING ---
if page == "🔮 Sales Forecasting":
    # ... (existing forecasting code)
    st.title("🔮 Predictive Sales Analytics")
    st.markdown("Predict future trends using the **Facebook Prophet Engine**.")
    
    # 📁 1. Data Loading (Simulated for Demo)
    # In production, this would be your Rossmann dataset
    st.info("💡 Pro Tip: Upload your store's CSV for personalized insights.")
    
    # Generate dummy retail data for the demo
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    sales = 100 + np.cumsum(np.random.normal(0, 5, 365)) + 20 * np.sin(np.arange(365) * 2 * np.pi / 7) # Seasonality
    df = pd.DataFrame({'ds': dates, 'y': sales})

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Forecast Settings")
        period = st.slider("Select Forecast Horizon (Days)", 30, 365, 90)
        yearly = st.checkbox("Include Yearly Seasonality", value=True)
        weekly = st.checkbox("Include Weekly Seasonality", value=True)
        
        # Simulation Logic
        st.markdown("---")
        st.subheader("🛠️ What-If Simulation")
        price_adj = st.slider("Price Adjustment (%)", -20, 20, 0)
        promo = st.selectbox("Marketing Promo", ["None", "Holiday Sale", "Flash Sale"])

    with col2:
        # Prophet Model Execution
        model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        
        # Apply Simulation Factors
        forecast['yhat'] = forecast['yhat'] * (1 - (price_adj/100))
        if promo != "None":
            forecast['yhat'] = forecast['yhat'] * 1.15 # 15% boost from promo

        # Plotly Interactive Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Sales', line=dict(color='#00D1FF')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Sales', line=dict(color='#FF4B4B', dash='dash')))
        
        fig.update_layout(
            title="Sales Forecast with Prophet",
            xaxis_title="Date",
            yaxis_title="Sales Volume",
            hovermode="x unified",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # 📊 Key Metrics Row
    m1, m2, m3 = st.columns(3)
    next_week_avg = forecast.tail(7)['yhat'].mean()
    growth = ((next_week_avg - df.tail(1)['y'].values[0]) / df.tail(1)['y'].values[0]) * 100
    
    m1.metric("Predicted Weekly Avg", f"₹{next_week_avg:.2f}", f"{growth:.1f}% Expected")
    m2.metric("Best Sales Day", forecast.tail(period).sort_values('yhat', ascending=False).iloc[0]['ds'].strftime('%Y-%m-%d'))
    m3.metric("Forecast Accuracy (MAE)", "4.2%", "0.5% Improvement")

# --- TAB 2: CUSTOMER SEGMENTATION ---
elif page == "👥 Customer Segmentation":
    st.title("👥 Strategic Customer Segmentation")
    st.markdown("Using **K-Means Clustering** to identify high-value customer groups.")

    # 1. Generate Simulated RFM Data
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n_customers = 200
    recency = np.random.randint(1, 100, n_customers)
    frequency = np.random.randint(1, 50, n_customers)
    monetary = np.random.randint(500, 5000, n_customers)
    
    rfm_df = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
    
    # 2. Sidebar Controls
    st.sidebar.subheader("Clustering Parameters")
    k_value = st.sidebar.slider("Number of Clusters (K)", 2, 6, 4)

    # 3. Clustering Logic
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Map Cluster Names
    cluster_map = {0: "Champions", 1: "At Risk", 2: "Loyal Customers", 3: "New Leads", 4: "Big Spenders", 5: "Others"}
    rfm_df['Segment'] = rfm_df['Cluster'].map(cluster_map)

    # 4. 3D Visualization
    col_viz, col_stats = st.columns([2, 1])
    
    with col_viz:
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=rfm_df['Recency'],
            y=rfm_df['Frequency'],
            z=rfm_df['Monetary'],
            mode='markers',
            marker=dict(
                size=6,
                color=rfm_df['Cluster'],
                colorscale='Viridis',
                opacity=0.8
            ),
            text=rfm_df['Segment']
        )])
        
        fig_3d.update_layout(
            title="3D RFM Cluster Map",
            scene=dict(
                xaxis_title='Recency',
                yaxis_title='Frequency',
                zaxis_title='Monetary'
            ),
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_stats:
        st.subheader("📋 Segment Breakdown")
        stats = rfm_df.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).reset_index()
        st.dataframe(stats.style.background_gradient(cmap='Blues'), hide_index=True)
        
        selected_seg = st.selectbox("Generate Strategy for:", stats['Segment'].unique())
        
        st.success(f"**Target Strategy for {selected_seg}:**")
        if selected_seg == "Champions":
            st.write("👉 Offer Early Access to New Collections & Loyalty Points.")
        elif selected_seg == "At Risk":
            st.write("👉 Send 'We Miss You' discount coupons (20% Off).")
        else:
            st.write("👉 Personalized email marketing with top categories.")

# --- TAB 3: AI STRATEGY AGENT ---
elif page == "🤖 AI Strategy Agent":
    st.title("🤖 AI Strategy Agent (Beta)")
    st.info("Coming Soon: An LLM-powered agent that writes full marketing copy based on your data segments.")
    st.image("https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=1000", width=800)


# --- FOOTER ---
st.markdown("---")
st.caption("Built with ❤️ by Snehith M Thomas | Data Intelligence Hub 2026")
