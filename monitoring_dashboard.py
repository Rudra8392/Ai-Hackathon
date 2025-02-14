import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def create_monitoring_dashboard():
    st.set_page_config(page_title="Oil & Gas Production Monitoring", layout="wide")
    st.title("Oil & Gas Production Monitoring Dashboard")

    # Sidebar for controls
    st.sidebar.header("Controls")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now())
    )

    # Create sample data (replace with real data in production)
    dates = pd.date_range(start=date_range[0], end=date_range[1], freq='H')
    n_samples = len(dates)

    data = pd.DataFrame({
        'Time': dates,
        'Total_Flow': np.random.normal(1000, 100, n_samples),
        'Oil_Flow': np.random.normal(700, 50, n_samples),
        'Water_Flow': np.random.normal(200, 30, n_samples),
        'Gas_Flow': np.random.normal(500, 80, n_samples),
        'Pressure': np.random.normal(200, 20, n_samples),
        'Temperature': np.random.normal(60, 5, n_samples)
    })

    # Create dashboard layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Production Rates")
        fig_production = go.Figure()
        fig_production.add_trace(go.Scatter(
            x=data['Time'],
            y=data['Total_Flow'],
            name='Total Flow',
            line=dict(color='blue')
        ))
        fig_production.add_trace(go.Scatter(
            x=data['Time'],
            y=data['Oil_Flow'],
            name='Oil Flow',
            line=dict(color='green')
        ))
        fig_production.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Flow Rate (BLPD)"
        )
        st.plotly_chart(fig_production, use_container_width=True)

    with col2:
        st.subheader("Pressure and Temperature")
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Scatter(
            x=data['Time'],
            y=data['Pressure'],
            name='Pressure',
            line=dict(color='red')
        ))
        fig_pt.add_trace(go.Scatter(
            x=data['Time'],
            y=data['Temperature'],
            name='Temperature',
            line=dict(color='orange'),
            yaxis="y2"
        ))
        fig_pt.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Pressure (bar)",
            yaxis2=dict(
                title="Temperature (°C)",
                overlaying="y",
                side="right"
            )
        )
        st.plotly_chart(fig_pt, use_container_width=True)

    # Key Performance Indicators (KPIs)
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Total Flow",
            f"{data['Total_Flow'].mean():.0f} BLPD",
            f"{data['Total_Flow'].pct_change().mean() * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Average Pressure",
            f"{data['Pressure'].mean():.1f} bar",
            f"{data['Pressure'].pct_change().mean() * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Water Cut",
            f"{(data['Water_Flow'] / data['Total_Flow']).mean() * 100:.1f}%"
        )

    with col4:
        st.metric(
            "GOR",
            f"{(data['Gas_Flow'] / data['Oil_Flow']).mean():.0f} scf/bbl"
        )

    # Historical Data Table
    st.subheader("Historical Data")
    st.dataframe(data.tail(10))


def main():
    create_monitoring_dashboard()


if __name__ == "__main__":
    main()
