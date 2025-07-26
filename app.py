import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Constants ---
st.set_page_config(layout="wide", page_title="AI Facebook Ads Analyzer")

REQUIRED_COLUMNS = [
    'Reporting starts', 'Reporting ends', 'Ad name', 'Ad delivery', 
    'Attribution setting', 'Results', 'Result indicator', 'Reach', 'Frequency', 
    'Cost per results', 'Ad set budget', 'Ad set budget type', 'Amount spent (USD)', 
    'Ends', 'Quality ranking', 'Engagement rate ranking', 'Conversion rate ranking', 
    'Impressions', 'CPM (cost per 1,000 impressions) (USD)', 'Link clicks', 
    'shop_clicks', 'CPC (cost per link click) (USD)', 'CTR (link click-through rate)', 
    'Clicks (all)', 'CTR (all)', 'CPC (all) (USD)'
]

# --- Helper Functions ---

def load_and_process_data(uploaded_file):
    """
    Loads data from an uploaded file, validates columns, and performs necessary preprocessing.
    """
    try:
        # Determine file type and read into pandas
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # --- 1. Strict Column Validation ---
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        st.error("CSV/Excel format not recognized. Please ensure your export contains all standard Facebook Ads columns.")
        st.info(f"Missing columns: {list(set(REQUIRED_COLUMNS) - set(df.columns))}")
        return None

    # --- 2. Data Type Conversion & Cleaning ---
    date_cols = ['Reporting starts', 'Reporting ends', 'Ends']
    date_format = '%Y-%m-%d'  # Adjust if your data uses a different format
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')

    numeric_cols = [
        'Results', 'Reach', 'Frequency', 'Cost per results', 'Amount spent (USD)', 
        'Impressions', 'CPM (cost per 1,000 impressions) (USD)', 'Link clicks', 
        'shop_clicks', 'CPC (cost per link click) (USD)', 'CTR (link click-through rate)', 
        'Clicks (all)', 'CTR (all)', 'CPC (all) (USD)'
    ]
    for col in numeric_cols:
        # Replace non-numeric placeholders like '-' before converting
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Handle specific columns like rankings which can have text values
    for col in ['Quality ranking', 'Engagement rate ranking', 'Conversion rate ranking']:
        # Keep as object/category type, replacing nulls with a placeholder
        df[col] = df[col].fillna('N/A')

    st.success("File successfully uploaded and processed!")
    return df

def display_kpis(df):
    """
    Calculates and displays key performance indicators using st.metric.
    """
    st.header("📊 Overall Performance KPIs")

    # --- KPI Calculations ---
    total_spend = df['Amount spent (USD)'].sum()
    total_impressions = df['Impressions'].sum()
    total_reach = df['Reach'].sum()
    total_link_clicks = df['Link clicks'].sum()
    total_clicks_all = df['Clicks (all)'].sum()
    total_results = df['Results'].sum()

    # Handle division by zero gracefully
    overall_ctr = (total_link_clicks / total_impressions * 100) if total_impressions > 0 else 0
    overall_cpc = (total_spend / total_link_clicks) if total_link_clicks > 0 else 0
    overall_cpm = (total_spend / total_impressions * 1000) if total_impressions > 0 else 0
    overall_cpr = (total_spend / total_results) if total_results > 0 else 0
    
    # --- KPI Display ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Total Spend (USD)", value=f"${total_spend:,.2f}")
        st.metric(label="Total Impressions", value=f"{total_impressions:,.0f}")
    with col2:
        st.metric(label="Total Reach", value=f"{total_reach:,.0f}")
        st.metric(label="Total Link Clicks", value=f"{total_link_clicks:,.0f}")
    with col3:
        st.metric(label="Overall CTR (%)", value=f"{overall_ctr:.2f}%")
        st.metric(label="Overall CPC (USD)", value=f"${overall_cpc:.2f}")
    with col4:
        st.metric(label="Overall CPM (USD)", value=f"${overall_cpm:.2f}")
        st.metric(label="Total Clicks (All)", value=f"{total_clicks_all:,.0f}")
    with col5:
        st.metric(label="Total Results", value=f"{total_results:,.0f}")
        st.metric(label="Cost Per Result (USD)", value=f"${overall_cpr:.2f}")

def generate_performance_charts(df):
    """
    Creates and displays interactive charts using Plotly Express.
    """
    st.header("📈 Performance Visualizations")

    # --- Time-Series Analysis ---
    st.subheader("Performance Over Time")
    
    date_axis = st.selectbox(
        "Choose date axis for time-series charts:", 
        ('Reporting starts', 'Reporting ends'), 
        key='date_axis_selector'
    )
    
    # Aggregate data by day
    daily_data = df.groupby(pd.Grouper(key=date_axis, freq='D')).agg({
        'Amount spent (USD)': 'sum',
        'Link clicks': 'sum',
        'Impressions': 'sum'
    }).reset_index()

    # Bar chart for Spend
    fig_spend = px.bar(daily_data, x=date_axis, y='Amount spent (USD)', title='Daily Spend Over Time',
                      labels={'Amount spent (USD)': 'Amount Spent (USD)', date_axis: 'Date'})
    st.plotly_chart(fig_spend, use_container_width=True)

    # Bar chart for Clicks and Impressions
    fig_clicks_impressions = px.bar(daily_data, x=date_axis, y=['Link clicks', 'Impressions'],
                                   barmode='group',
                                   title='Daily Link Clicks and Impressions',
                                   labels={'value': 'Count', date_axis: 'Date'})
    st.plotly_chart(fig_clicks_impressions, use_container_width=True)

    # --- Heatmap of Daily Spend ---
    st.subheader("Heatmap: Daily Spend")
    heatmap_data = daily_data.copy()
    heatmap_data['day'] = heatmap_data[date_axis].dt.day
    heatmap_data['month'] = heatmap_data[date_axis].dt.month
    pivot = heatmap_data.pivot_table(index='month', columns='day', values='Amount spent (USD)', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Spend (USD)'})
    plt.xlabel('Day')
    plt.ylabel('Month')
    st.pyplot(fig)

    # --- Breakdown Bar Charts ---
    st.subheader("Performance Breakdown")
    
    breakdown_dim = st.selectbox(
        "Select dimension to break down performance:", 
        ('Ad name', 'Ad delivery'), 
        key='breakdown_selector'
    )

    # Group by the selected dimension
    breakdown_data = df.groupby(breakdown_dim).agg({
        'Amount spent (USD)': 'sum',
        'Link clicks': 'sum',
        'Impressions': 'sum',
        'Results': 'sum'
    }).reset_index()
    
    # Calculate metrics for breakdown view
    breakdown_data['CTR (link click-through rate)'] = (breakdown_data['Link clicks'] / breakdown_data['Impressions'] * 100).fillna(0)
    breakdown_data['Cost per results'] = (breakdown_data['Amount spent (USD)'] / breakdown_data['Results']).fillna(0)

    col1, col2 = st.columns(2)
    
    with col1:
        fig_spend_breakdown = px.bar(breakdown_data, x=breakdown_dim, y='Amount spent (USD)',
                                     title=f'Amount Spent by {breakdown_dim}',
                                     labels={'Amount spent (USD)': 'Amount Spent (USD)'})
        st.plotly_chart(fig_spend_breakdown, use_container_width=True)
        
        fig_cpr_breakdown = px.bar(breakdown_data, x=breakdown_dim, y='Cost per results',
                                     title=f'Cost per Result by {breakdown_dim}',
                                     labels={'Cost per results': 'Cost per Result (USD)'})
        st.plotly_chart(fig_cpr_breakdown, use_container_width=True)

    with col2:
        fig_ctr_breakdown = px.bar(breakdown_data, x=breakdown_dim, y='CTR (link click-through rate)',
                                     title=f'Link CTR by {breakdown_dim}',
                                     labels={'CTR (link click-through rate)': 'Link CTR (%)'})
        st.plotly_chart(fig_ctr_breakdown, use_container_width=True)

    # --- Scatter Plot: CPC vs. CTR by Ad ---
    st.subheader("Scatter Plot: CPC vs. CTR by Ad")
    if 'CPC (cost per link click) (USD)' in df.columns and 'CTR (link click-through rate)' in df.columns:
        scatter_df = df.groupby('Ad name').agg({
            'CPC (cost per link click) (USD)': 'mean',
            'CTR (link click-through rate)': 'mean',
            'Amount spent (USD)': 'sum',
            'Impressions': 'sum'
        }).reset_index()
        fig_scatter = px.scatter(
            scatter_df, x='CPC (cost per link click) (USD)', y='CTR (link click-through rate)',
            size='Amount spent (USD)', color='Impressions',
            hover_name='Ad name',
            title='CPC vs. CTR by Ad (Bubble size: Spend, Color: Impressions)',
            labels={'CPC (cost per link click) (USD)': 'CPC (USD)', 'CTR (link click-through rate)': 'CTR (%)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Funnel Visualization ---
    st.subheader("Funnel: Impressions → Clicks → Results")
    funnel_vals = [df['Impressions'].sum(), df['Link clicks'].sum(), df['Results'].sum()]
    funnel_labels = ['Impressions', 'Link Clicks', 'Results']
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_labels,
        x=funnel_vals,
        textinfo="value+percent initial"
    ))
    fig_funnel.update_layout(title="Funnel: Impressions to Results")
    st.plotly_chart(fig_funnel, use_container_width=True)

    # --- Stacked Area Chart: Spend Over Time by Ad ---
    st.subheader("Stacked Area Chart: Spend Over Time by Ad")
    if 'Ad name' in df.columns:
        area_df = df.groupby([date_axis, 'Ad name'])['Amount spent (USD)'].sum().reset_index()
        area_pivot = area_df.pivot(index=date_axis, columns='Ad name', values='Amount spent (USD)').fillna(0)
        fig_area = go.Figure()
        for ad in area_pivot.columns:
            fig_area.add_trace(go.Scatter(
                x=area_pivot.index, y=area_pivot[ad],
                mode='lines', stackgroup='one', name=ad
            ))
        fig_area.update_layout(title="Spend Over Time by Ad", xaxis_title="Date", yaxis_title="Spend (USD)")
        st.plotly_chart(fig_area, use_container_width=True)

    # --- Pie Chart: Spend by Ad Delivery ---
    st.subheader("Pie Chart: Spend by Ad Delivery")
    if 'Ad delivery' in df.columns:
        pie_df = df.groupby('Ad delivery')['Amount spent (USD)'].sum().reset_index()
        fig_pie = px.pie(pie_df, names='Ad delivery', values='Amount spent (USD)',
                         title='Spend by Ad Delivery')
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Box Plot: Distribution of CPC by Ad Delivery ---
    st.subheader("Box Plot: CPC Distribution by Ad Delivery")
    if 'CPC (cost per link click) (USD)' in df.columns and 'Ad delivery' in df.columns:
        fig_box = px.box(df, x='Ad delivery', y='CPC (cost per link click) (USD)',
                        title='CPC Distribution by Ad Delivery',
                        labels={'CPC (cost per link click) (USD)': 'CPC (USD)', 'Ad delivery': 'Ad Delivery'})
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Ranking Table with Conditional Formatting ---
    st.subheader("Ranking Table: Ads by Cost per Result")
    if 'Ad name' in df.columns and 'Cost per results' in df.columns:
        rank_df = df.groupby('Ad name').agg({
            'Cost per results': 'mean',
            'Amount spent (USD)': 'sum',
            'Results': 'sum',
            'Impressions': 'sum'
        }).reset_index().sort_values('Cost per results')
        st.dataframe(rank_df.style.background_gradient(cmap='RdYlGn_r', subset=['Cost per results']))

    # --- Correlation Matrix ---
    st.subheader("Correlation Matrix of Numeric Metrics")
    numeric_cols = [
        'Results', 'Reach', 'Frequency', 'Cost per results', 'Amount spent (USD)',
        'Impressions', 'CPM (cost per 1,000 impressions) (USD)', 'Link clicks',
        'shop_clicks', 'CPC (cost per link click) (USD)', 'CTR (link click-through rate)',
        'Clicks (all)', 'CTR (all)', 'CPC (all) (USD)'
    ]
    corr_df = df[numeric_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    plt.title("Correlation Matrix of Numeric Metrics")
    st.pyplot(fig_corr)


def get_ai_recommendations(api_key, df):
    """
    Generates a data summary and calls OpenAI API for actionable recommendations.
    """
    st.header("🤖 AI-Driven Recommendations")
    
    if st.button("Generate AI Recommendations"):
        if not api_key:
            st.warning("Please enter your OpenAI API Key to generate recommendations.")
            return

        with st.spinner("🧠 Analyzing data and generating insights..."):
            try:
                client = OpenAI(api_key=api_key)

                # --- Create a concise data summary for the AI prompt ---
                df_summary = df.head().to_string()
                kpis = {
                    "Total Spend": f"${df['Amount spent (USD)'].sum():,.2f}",
                    "Total Impressions": f"{df['Impressions'].sum():,.0f}",
                    "Total Link Clicks": f"{df['Link clicks'].sum():,.0f}",
                    "Overall CPC": f"${(df['Amount spent (USD)'].sum() / df['Link clicks'].sum()):.2f}" if df['Link clicks'].sum() > 0 else "N/A",
                    "Overall CTR": f"{(df['Link clicks'].sum() / df['Impressions'].sum() * 100):.2f}%" if df['Impressions'].sum() > 0 else "N/A",
                    "Total Results": f"{df['Results'].sum():,.0f}",
                    "Overall Cost per Result": f"${(df['Amount spent (USD)'].sum() / df['Results'].sum()):.2f}" if df['Results'].sum() > 0 else "N/A"
                }
                
                # Identify top and bottom performers by a key metric
                ad_perf = df.groupby('Ad name')['Cost per results'].mean().sort_values()
                top_performer = ad_perf.head(1).index[0] if not ad_perf.empty else "N/A"
                bottom_performer = ad_perf.tail(1).index[0] if not ad_perf.empty else "N/A"

                prompt = f"""
                You are an expert Facebook Ads data analyst tasked with providing optimization advice.
                Analyze the following data summary and provide actionable recommendations.

                **Data Context:**
                - Reporting Period: {df['Reporting starts'].min().strftime('%Y-%m-%d')} to {df['Reporting ends'].max().strftime('%Y-%m-%d')}
                - Data Preview (first 5 rows):
                {df_summary}

                **Key Performance Indicators (KPIs):**
                {kpis}

                **Performance Highlights:**
                - Top performing ad by Cost per Result: '{top_performer}'
                - Bottom performing ad by Cost per Result: '{bottom_performer}'

                **Your Task:**
                Based on this data, provide 3-5 actionable, specific, and concise recommendations to improve campaign performance.
                Focus on optimizing budget, improving creative/ad copy, A/B testing suggestions, and identifying areas for further investigation.
                Structure your response in Markdown. Start with a brief summary of your findings, followed by a numbered list of recommendations.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini", # Using a cost-effective and capable model
                    messages=[
                        {"role": "system", "content": "You are an expert Facebook Ads data analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Could not get recommendations from AI. Error: {e}")
                st.info("Please check your API key and network connection.")


# --- Natural Language Q&A Function ---
def data_qa_section(api_key, df):
    """
    Allows the user to ask natural language questions about the uploaded data and get answers from OpenAI.
    """
    st.header("💬 Ask Questions About Your Data")
    st.info("Type a question about your Facebook Ads data (e.g., 'Which ad had the highest CTR?', 'What was the average spend per day?').", icon="❓")
    user_question = st.text_input("Ask a question about your data:", key="data_qa_input")
    if st.button("Get Answer", key="data_qa_button"):
        if not api_key:
            st.warning("Please enter your OpenAI API Key to use the Q&A feature.")
            return
        if not user_question.strip():
            st.warning("Please enter a question.")
            return
        with st.spinner("Thinking..."):
            try:
                client = OpenAI(api_key=api_key)
                # Provide a concise data sample and columns for context
                df_sample = df.head(10).to_string()
                columns = ', '.join(df.columns)
                prompt = f"""
You are a data analyst. The user uploaded a Facebook Ads dataset with the following columns: {columns}.
Here is a sample of the data (first 10 rows):
{df_sample}

Answer the following question about the data. If you need to calculate something, explain your reasoning and show the answer clearly.
Question: {user_question}
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst for Facebook Ads data."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Could not get an answer from AI. Error: {e}")
                st.info("Please check your API key and network connection.")


# --- Main Application ---

def main():
    # --- Custom CSS for Modern Look ---
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stMetric {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 1.2em 0.5em 1.2em 0.5em;
            margin-bottom: 0.5em;
        }
        .stDataFrame, .stTable {
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .section-card {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            padding: 2rem 2rem 1.5rem 2rem;
            margin-bottom: 2rem;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 0.7rem;
        }
        .section-subtitle {
            font-size: 1.1rem;
            color: #4a5568;
            margin-bottom: 1.2rem;
        }
        .stButton>button {
            border-radius: 8px;
            background: #2563eb;
            color: #fff;
            font-weight: 600;
        }
        .stTextInput>div>div>input[type='password'] {
            background: #f1f5f9;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Sidebar Navigation ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/5968/5968764.png", width=60)
    st.sidebar.title("Facebook Ads Analyzer")
    st.sidebar.markdown("""
        <span style='font-size:1.1em;'>Upload your Facebook Ads export and explore your data visually and interactively.</span>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Navigation**")
    nav = st.sidebar.selectbox(
        "Go to section:",
        ["Upload & Filter", "KPIs & Overview", "Visualizations", "AI Insights", "Q&A"],
        index=0,
        key="sidebar_nav_dropdown"
    )

    # --- Main Title ---
    st.markdown("""
        <div style='text-align:center; margin-bottom:2rem;'>
            <h1 style='font-size:2.5rem; color:#2563eb; margin-bottom:0.2em;'>🚀 AI-Powered Facebook Ads Data Analyzer</h1>
            <div style='font-size:1.2rem; color:#4a5568;'>Upload your raw Facebook Ads export file (<code>.csv</code> or <code>.xlsx</code>) to get started.<br>This tool validates your data, displays key metrics, visualizes performance, and provides AI-driven optimization tips.</div>
        </div>
    """, unsafe_allow_html=True)

    # --- File Upload & Data Filtering Section ---
    if nav == "Upload & Filter":
        with st.container():
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>1. Upload & Filter Your Data</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose a Facebook Ads export file", 
                type=['csv', 'xlsx']
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if uploaded_file is not None:
                df = load_and_process_data(uploaded_file)
                st.session_state['df'] = df
    else:
        df = st.session_state.get('df', None)

    # --- Data Filtering Card (if data loaded) ---
    if nav == "Upload & Filter" and st.session_state.get('df', None) is not None:
        df = st.session_state['df']
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>2. Filter Options</div>", unsafe_allow_html=True)
        min_date = df['Reporting starts'].min()
        max_date = df['Reporting ends'].max()
        date_range = st.date_input(
            "Select reporting date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range_filter"
        )
        ad_names = sorted(df['Ad name'].unique())
        selected_ads = st.multiselect(
            "Filter by Ad Name (optional):",
            options=ad_names,
            default=ad_names,
            key="ad_name_filter"
        )
        campaign_col = None
        for col in ['Campaign name', 'Ad set name', 'Ad set', 'Campaign']:
            if col in df.columns:
                campaign_col = col
                break
        if campaign_col:
            campaign_names = sorted(df[campaign_col].unique())
            selected_campaigns = st.multiselect(
                f"Filter by {campaign_col} (optional):",
                options=campaign_names,
                default=campaign_names,
                key="campaign_filter"
            )
        else:
            selected_campaigns = None
        filtered_df = df.copy()
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_format = '%Y-%m-%d'
            filtered_df = filtered_df[
                (filtered_df['Reporting starts'] >= pd.to_datetime(date_range[0], format=date_format, errors='coerce')) &
                (filtered_df['Reporting ends'] <= pd.to_datetime(date_range[1], format=date_format, errors='coerce'))
            ]
        if selected_ads:
            filtered_df = filtered_df[filtered_df['Ad name'].isin(selected_ads)]
        if campaign_col and selected_campaigns:
            filtered_df = filtered_df[filtered_df[campaign_col].isin(selected_campaigns)]
        st.success(f"Filtered data: {len(filtered_df)} rows (of {len(df)})")
        st.dataframe(filtered_df.head(20))
        st.session_state['filtered_df'] = filtered_df
        st.markdown("</div>", unsafe_allow_html=True)

    # --- KPIs & Overview Section ---
    if nav == "KPIs & Overview" and st.session_state.get('filtered_df', None) is not None:
        filtered_df = st.session_state['filtered_df']
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Key Performance Indicators</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>A quick glance at your most important metrics.</div>", unsafe_allow_html=True)
        display_kpis(filtered_df)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Visualizations Section ---
    if nav == "Visualizations" and st.session_state.get('filtered_df', None) is not None:
        filtered_df = st.session_state['filtered_df']
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Visual Data Explorer</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Explore your data with interactive and insightful charts.</div>", unsafe_allow_html=True)
        generate_performance_charts(filtered_df)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- AI Insights Section ---
    if nav == "AI Insights" and st.session_state.get('filtered_df', None) is not None:
        filtered_df = st.session_state['filtered_df']
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🤖 AI-Powered Recommendations</div>", unsafe_allow_html=True)
        st.info("Provide your OpenAI API key to unlock AI-driven insights. Your key is not stored.", icon="🔒")
        api_key_input = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="You can find your API key on the OpenAI dashboard. For deployed apps, use st.secrets. Leave blank to use the key from .env file.",
            key="openai_api_key_input"
        )
        api_key = api_key_input or os.getenv("OPENAI_API_KEY")
        get_ai_recommendations(api_key, filtered_df)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Q&A Section ---
    if nav == "Q&A" and st.session_state.get('filtered_df', None) is not None:
        filtered_df = st.session_state['filtered_df']
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💬 Ask Questions About Your Data</div>", unsafe_allow_html=True)
        api_key_input_qa = st.text_input(
            "OpenAI API Key (for Q&A)",
            type="password",
            help="You can use the same key as above or leave blank to use the .env key.",
            key="openai_api_key_input_qa"
        )
        api_key_qa = api_key_input_qa or os.getenv("OPENAI_API_KEY")
        data_qa_section(api_key_qa, filtered_df)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()