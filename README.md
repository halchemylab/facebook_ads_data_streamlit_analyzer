# 🚀 AI-Powered Facebook Ads Data Analyzer

Welcome to the AI-Powered Facebook Ads Data Analyzer! This Streamlit application is designed to help you analyze your Facebook Ads performance data, visualize key metrics, and generate actionable recommendations using the power of AI.

## ✨ Features

- **📊 Interactive KPI Dashboard:** Get a comprehensive overview of your campaign performance with key metrics like Total Spend, Impressions, Reach, Link Clicks, CTR, CPC, and more.
- **📈 Performance Visualizations:** Dive deeper into your data with interactive charts that visualize performance over time and break it down by different dimensions like Ad Name and Ad Delivery.
- **🤖 AI-Driven Recommendations:** Unlock actionable insights and optimization tips for your campaigns by leveraging the power of OpenAI's GPT models.
- **📄 Data Validation:** The application automatically validates your uploaded data to ensure it contains the required columns and handles data type conversions for accurate analysis.
- **📁 Support for Multiple File Types:** Upload your Facebook Ads export data as a `.csv` or `.xlsx` file.

##  How to Use

1.  **Upload Your Data:** Click on the "Choose a Facebook Ads export file" button to upload your data file.
2.  **View KPIs and Visualizations:** Once the data is uploaded and processed, you can view the overall performance KPIs and interactive charts.
3.  **Generate AI Recommendations:** Expand the "Get AI-Powered Recommendations" section, enter your OpenAI API key, and click the "Generate AI Recommendations" button to receive optimization tips.

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
- **Plotly Express:** For creating interactive visualizations.
- **OpenAI API:** For generating AI-powered recommendations.

## 🚀 Running the Application Locally

To run this application on your local machine, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/facebook-ads-data-streamlit-analyzer.git
    cd facebook-ads-data-streamlit-analyzer
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**

    ```bash
    streamlit run app.py
    ```
