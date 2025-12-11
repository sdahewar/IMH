"""
IndiaMART Insights Engine - Interactive Dashboard
Streamlit-based visualization for call analytics

Run with: streamlit run app.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="IndiaMART Insights Engine",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .recommendation-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_raw_data():
    """Load the raw Excel data"""
    paths = ["Data Voice Hackathon_Master.xlsx", "data/Data Voice Hackathon_Master.xlsx"]
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_excel(path)
            except:
                pass
    return None


@st.cache_data
def extract_quick_insights(df):
    """Extract insights from existing summary column"""
    insights = {
        'sentiment': Counter(),
        'key_topics': Counter(),
        'alerts': 0
    }
    
    for _, row in df.iterrows():
        summary = str(row.get('summary', ''))
        
        if '@@@Sentiment:' in summary:
            try:
                sentiment_part = summary.split('@@@Sentiment:')[1].split('@@@')[0]
                if 'Positive' in sentiment_part:
                    insights['sentiment']['Positive'] += 1
                elif 'Negative' in sentiment_part:
                    insights['sentiment']['Negative'] += 1
                else:
                    insights['sentiment']['Neutral'] += 1
            except:
                pass
        
        if '@@@Key Topics:' in summary:
            try:
                topics_part = summary.split('@@@Key Topics:')[1].split('\n')[1]
                topics = [t.strip().lower() for t in topics_part.split(',')]
                for topic in topics:
                    if topic and len(topic) > 2:
                        insights['key_topics'][topic] += 1
            except:
                pass
        
        if 'Alert (If Any):' in summary:
            alert_part = summary.split('Alert (If Any):')[1].split('@@@')[0]
            if 'None' not in alert_part and alert_part.strip():
                insights['alerts'] += 1
    
    return insights


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_customer_type_distribution(df):
    counts = df['customer_type'].value_counts().head(10)
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        title="Customer Type Distribution",
        labels={'x': 'Number of Calls', 'y': 'Customer Type'},
        color=counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def plot_city_distribution(df):
    counts = df['city_name'].value_counts().head(15)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title="Top 15 Cities by Call Volume",
        labels={'x': 'City', 'y': 'Number of Calls'},
        color=counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    return fig


def plot_sentiment_pie(insights):
    if not insights['sentiment']:
        return None
    fig = px.pie(
        values=list(insights['sentiment'].values()),
        names=list(insights['sentiment'].keys()),
        title="Sentiment Distribution",
        color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plot_key_topics_wordcloud(insights):
    if not insights['key_topics']:
        return None
    top_topics = dict(insights['key_topics'].most_common(15))
    fig = px.bar(
        x=list(top_topics.values()),
        y=list(top_topics.keys()),
        orientation='h',
        title="Top 15 Key Topics from Calls",
        labels={'x': 'Frequency', 'y': 'Topic'},
        color=list(top_topics.values()),
        color_continuous_scale='Oranges'
    )
    fig.update_layout(showlegend=False, height=500)
    return fig


def plot_repeat_ticket_analysis(df):
    repeat_by_type = pd.crosstab(df['customer_type'], df['is_ticket_repeat60d'], normalize='index') * 100
    if 'Yes' in repeat_by_type.columns:
        repeat_by_type = repeat_by_type.sort_values('Yes', ascending=True)
        fig = px.bar(
            x=repeat_by_type['Yes'].values,
            y=repeat_by_type.index,
            orientation='h',
            title="Repeat Ticket Rate by Customer Type (%)",
            labels={'x': 'Repeat Rate %', 'y': 'Customer Type'},
            color=repeat_by_type['Yes'].values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=500)
        return fig
    return None


def plot_call_duration_distribution(df):
    fig = px.histogram(
        df,
        x='call_duration',
        nbins=50,
        title="Call Duration Distribution (seconds)",
        labels={'call_duration': 'Duration (seconds)', 'count': 'Number of Calls'},
        color_discrete_sequence=['#2d5a87']
    )
    fig.update_layout(height=400)
    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">📞 IndiaMART Insights Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Voice Call Analytics Dashboard | Powered by NVIDIA NIM (Nemotron-4-Mini-Hindi)</p>', unsafe_allow_html=True)
    
    df = load_raw_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure 'Data Voice Hackathon_Master.xlsx' is in the current directory.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    customer_types = ['All'] + list(df['customer_type'].unique())
    selected_type = st.sidebar.selectbox("Customer Type", customer_types)
    
    top_cities = ['All'] + list(df['city_name'].value_counts().head(20).index)
    selected_city = st.sidebar.selectbox("City", top_cities)
    
    repeat_filter = st.sidebar.radio("Repeat Ticket", ['All', 'Yes', 'No'])
    direction_filter = st.sidebar.radio("Call Direction", ['All', 'Incoming', 'Outgoing'])
    
    # Apply filters
    filtered_df = df.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['customer_type'] == selected_type]
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city_name'] == selected_city]
    if repeat_filter != 'All':
        filtered_df = filtered_df[filtered_df['is_ticket_repeat60d'] == repeat_filter]
    if direction_filter != 'All':
        filtered_df = filtered_df[filtered_df['FLAG_IN_OUT'] == direction_filter]
    
    insights = extract_quick_insights(filtered_df)
    
    # KEY METRICS
    st.header("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Calls", f"{len(filtered_df):,}")
    with col2:
        st.metric("Unique Customers", f"{filtered_df['glid'].nunique():,}")
    with col3:
        repeat_rate = len(filtered_df[filtered_df['is_ticket_repeat60d'] == 'Yes']) / len(filtered_df) * 100
        st.metric("Repeat Ticket Rate", f"{repeat_rate:.1f}%")
    with col4:
        st.metric("Avg Call Duration", f"{filtered_df['call_duration'].mean():.0f}s")
    with col5:
        st.metric("Alert Calls", f"{insights['alerts']:,}")
    
    # VISUALIZATIONS
    st.header("📈 Analytics")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Geographic", "👥 Customer Segments", "💬 Call Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_sentiment_pie(insights)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_key_topics_wordcloud(insights)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.plotly_chart(plot_city_distribution(filtered_df), use_container_width=True)
        
        st.subheader("City-wise Breakdown")
        city_stats = filtered_df.groupby('city_name').agg({
            'click_to_call_id': 'count',
            'call_duration': 'mean',
            'is_ticket_repeat60d': lambda x: (x == 'Yes').sum()
        }).round(2)
        city_stats.columns = ['Total Calls', 'Avg Duration', 'Repeat Tickets']
        city_stats = city_stats.sort_values('Total Calls', ascending=False).head(15)
        st.dataframe(city_stats, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_customer_type_distribution(filtered_df), use_container_width=True)
        with col2:
            fig = plot_repeat_ticket_analysis(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.plotly_chart(plot_call_duration_distribution(filtered_df), use_container_width=True)
    
    # ACTIONABLE INSIGHTS
    st.header("💡 Actionable Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Top Recommendation")
        st.markdown(f"""
        **Reduce Repeat Ticket Rate**
        
        Current repeat rate is **{repeat_rate:.1f}%**. Focus on:
        - First-call resolution training
        - Post-call verification
        - Issue categorization improvement
        
        **Potential Impact:** Reducing by 10% could save ~{int(len(filtered_df) * 0.1 * 0.37):,} follow-up calls
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 📌 Key Observations")
        top_topics = list(insights['key_topics'].most_common(5))
        st.markdown("**Top Issues Mentioned:**")
        for topic, count in top_topics:
            st.markdown(f"- {topic}: {count} mentions")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SAMPLE TRANSCRIPTS VIEWER
    st.header("📝 Sample Call Viewer")
    sample_idx = st.slider("Select Call", 0, min(100, len(filtered_df)-1), 0)
    
    if len(filtered_df) > sample_idx:
        sample_row = filtered_df.iloc[sample_idx]
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Call Details:**")
            st.write(f"📍 City: {sample_row['city_name']}")
            st.write(f"👤 Customer Type: {sample_row['customer_type']}")
            st.write(f"📞 Direction: {sample_row['FLAG_IN_OUT']}")
            st.write(f"⏱️ Duration: {sample_row['call_duration']}s")
            st.write(f"🔄 Repeat: {sample_row['is_ticket_repeat60d']}")
        
        with col2:
            st.markdown("**Call Summary:**")
            summary = sample_row['summary'][:1000] if pd.notna(sample_row['summary']) else "No summary"
            st.text_area("", value=summary, height=200, disabled=True)
        
        with st.expander("View Full Transcript"):
            transcript = sample_row['transcript'] if pd.notna(sample_row['transcript']) else "No transcript"
            st.text_area("", value=transcript, height=400, disabled=True)
    
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #888;">IndiaMART Insights Engine | Data Voice Hackathon 2024</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

