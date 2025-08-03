# 🚀 AI-Powered Facebook Ads Data Analyzer

Welcome to the AI-Powered Facebook Ads Data Analyzer! This Streamlit application is designed to help you analyze your Facebook Ads performance data, visualize key metrics, and generate actionable recommendations using the power of AI.


## ✨ Features

- **📊 Interactive KPI Dashboard:** Get a comprehensive overview of your campaign performance with key metrics like Total Spend, Impressions, Reach, Link Clicks, CTR, CPC, and more.
- **📈 Advanced Performance Visualizations:** Dive deeper into your data with a wide array of interactive charts, including:
    - Time-series line and bar charts
    - Heatmaps for daily spend analysis
    - Breakdown bar charts by ad name and delivery
    - Scatter plots to compare CPC vs. CTR
    - Funnel charts to visualize the conversion process
    - Stacked area charts for spend over time by ad
    - Pie charts for spend distribution
    - Box plots for CPC distribution
    - Correlation matrices to identify relationships between metrics
- **🤖 AI-Driven Recommendations:** Unlock actionable insights and optimization tips for your campaigns by leveraging the power of OpenAI's GPT models.
- **❓ Natural Language Q&A:** Ask questions about your uploaded data in plain English (e.g., "Which ad had the highest CTR?" or "What was the average spend per day?") and get instant answers powered by AI.
- **📄 Data Validation:** The application automatically validates your uploaded data to ensure it contains the required columns and handles data type conversions for accurate analysis.
- **📁 Multi-File Upload:** Upload and analyze multiple Facebook Ads export files (`.csv` or `.xlsx`) at once. The app will combine them for a holistic view.
- **⚔️ Comparison Mode:** Compare two different ads, campaigns, or source files side-by-side to easily identify performance differences.
- **Advanced Filtering:** Filter your data by date range, source file, ad name, and campaign name to focus on specific segments of your data.


## 🚀 How to Use

1.  **Navigate to the "Upload & Filter" Section:** Use the sidebar to navigate to the "Upload & Filter" section.
2.  **Upload Your Data:** Click on the "Choose one or more Facebook Ads export files" button to upload your data file(s).
3.  **Filter Your Data:** Use the advanced filtering options to narrow down your data by date range, source file, ad name, or campaign.
4.  **Explore the Sections:** Use the sidebar to navigate between the different sections of the app:
    - **KPIs & Overview:** Get a high-level overview of your campaign performance.
    - **Visualizations:** Explore your data with a wide range of interactive charts.
    - **AI Insights:** Generate AI-powered recommendations to optimize your campaigns.
    - **Q&A:** Ask natural language questions about your data.

## 📋 Data Requirements

To use this application, your export file must contain the following columns:

- `Reporting starts`
- `Reporting ends`
- `Ad name`
- `Ad delivery`
- `Attribution setting`
- `Results`
- `Result indicator`
- `Reach`
- `Frequency`
- `Cost per results`
- `Ad set budget`
- `Ad set budget type`
- `Amount spent (USD)`
- `Ends`
- `Quality ranking`
- `Engagement rate ranking`
- `Conversion rate ranking`
- `Impressions`
- `CPM (cost per 1,000 impressions) (USD)`
- `Link clicks`
- `shop_clicks`
- `CPC (cost per link click) (USD)`
- `CTR (link click-through rate)`
- `Clicks (all)`
- `CTR (all)`
- `CPC (all) (USD)`

## 🛠️ Technical Stack

- **Streamlit:** For creating the interactive web application.
- **Pandas:** For data manipulation and analysis.
- **Plotly Express & Plotly Graph Objects:** For creating interactive visualizations.
- **Seaborn & Matplotlib:** For creating heatmaps and correlation matrices.
- **OpenAI API:** For generating AI-powered recommendations and answering questions.
- **Dotenv:** For managing environment variables.

## 🚀 Running the Application Locally

To run this application on your local machine, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/facebook-ads-data-streamlit-analyzer.git
    cd facebook-ads-data-streamlit-analyzer
    ```

2.  **Install Dependencies:**

    Make sure you have Python 3.8+ installed. Then, run the following command to install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables:**

    Create a `.env` file in the root directory of the project and add your OpenAI API key:

    ```
    OPENAI_API_KEY="your-api-key"
    ```

4.  **Run the Application:**

    ```bash
    streamlit run app.py
    ```
