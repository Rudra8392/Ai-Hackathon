import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import joblib

def load_data():
    # Load preprocessed data
    data = pd.read_csv('preprocessed_data.csv')
    data['Time'] = pd.to_datetime(data['Time'])
    
    # Calculate GOR (assuming Gas_Flow is in standard cubic feet and Oil_Flow is in barrels)
    if 'Gas_Flow' in data.columns and 'Oil_Flow' in data.columns:
        data['GOR'] = data['Gas_Flow'] / data['Oil_Flow']
    else:
        # If Gas_Flow is not available, we'll use a dummy calculation for demonstration
        # In a real scenario, you'd need actual gas flow data
        data['GOR'] = data['Total_Flow'] / data['Oil_Flow'] * 1000  # Dummy calculation

    return data

def create_monitoring_dashboard():
    st.set_page_config(page_title="Oil & Gas Production Monitoring", layout="wide")
    st.title("Oil & Gas Production Monitoring Dashboard")
    
    # Load data
    data = load_data()
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(data['Time'].min().date(), data['Time'].max().date())
    )
    
    # Filter data based on date range
    mask = (data['Time'].dt.date >= date_range[0]) & (data['Time'].dt.date <= date_range[1])
    filtered_data = data.loc[mask]
    
    # Create dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Production Rates")
        fig_production = go.Figure()
        fig_production.add_trace(go.Scatter(
            x=filtered_data['Time'], 
            y=filtered_data['Total_Flow'],
            name='Total Flow',
            line=dict(color='blue')
        ))
        fig_production.add_trace(go.Scatter(
            x=filtered_data['Time'], 
            y=filtered_data['Oil_Flow'],
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
            x=filtered_data['Time'], 
            y=filtered_data['Pressure'],
            name='Pressure',
            line=dict(color='red')
        ))
        fig_pt.add_trace(go.Scatter(
            x=filtered_data['Time'], 
            y=filtered_data['Temperature'],
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
            f"{filtered_data['Total_Flow'].mean():.0f} BLPD",
            f"{filtered_data['Total_Flow'].pct_change().mean()*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Average Pressure",
            f"{filtered_data['Pressure'].mean():.1f} bar",
            f"{filtered_data['Pressure'].pct_change().mean()*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Water Cut",
            f"{(filtered_data['Water_Flow']/filtered_data['Total_Flow']).mean()*100:.1f}%"
        )
    
    with col4:
        st.metric(
            "GOR",
            f"{filtered_data['GOR'].mean():.0f} scf/bbl"
        )
    
    # GOR Chart
    st.subheader("Gas-Oil Ratio (GOR) Over Time")
    fig_gor = go.Figure()
    fig_gor.add_trace(go.Scatter(
        x=filtered_data['Time'],
        y=filtered_data['GOR'],
        name='GOR',
        line=dict(color='purple')
    ))
    fig_gor.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="GOR (scf/bbl)"
    )
    st.plotly_chart(fig_gor, use_container_width=True)
    
    
    # Historical Data Table
    st.subheader("Historical Data")
    st.dataframe(filtered_data.tail(10))

def main():
    create_monitoring_dashboard()

if __name__ == "__main__":
    main()

