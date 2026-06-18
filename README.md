# System Capacity & Care Load Analytics for Unaccompanied Children 📊

![GitHub stars](https://img.shields.io/github/stars/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children?style=social)
![GitHub forks](https://img.shields.io/github/forks/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children?style=social)
![GitHub license](https://img.shields.io/github/license/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children)

This project provides a comprehensive dashboard for analyzing system capacity and care load related to the Unaccompanied Alien Children (UAC) Program. It leverages data visualization and predictive modeling to offer insights into the operational dynamics of child welfare services.

## 🔗 Important links

*   **GitHub Repository**: [System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children](https://Prathamesh666.github.io/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children)
*   **Research Paper**: [Link to Research Paper](https://prathamesh666.github.io/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children/Research%20Paper.html)
*  **Streamlit dashboard**:[System Capacity & Care Load Analytics for Unaccompanied Children](https://um-ussystemcapacitycareloadanalyticsforunaccompaniedchildren.streamlit.app/)

## 🚀 Features

*   **Interactive Dashboard**: A Streamlit-based application offering a rich user experience with filtering, toggles, and detailed visualizations.
*   **Data Ingestion & Validation**: Robust handling of CSV uploads, including schema validation and data cleaning.
*   **Key Performance Indicators (KPIs)**: Real-time display of critical metrics such as total children under care, average system load, positive net intake days, and more.
*   **Detailed Trend Analysis**: Visualizations for daily, weekly, and monthly trends, including 3D representations of complex data.
*   **Pressure & Stress Identification**: Tools to identify operational strain through metrics like the Operational Pressure Index (OPI) and detected strain windows.
*   **Predictive Modeling**: Implementation of regression and classification models (e.g., Gradient Boosting, XGBoost) for forecasting and load category prediction.
*   **Feature Engineering**: Creation of advanced features (lags, rolling statistics, cyclical encodings) to enhance model accuracy.
*   **Forecasting**: Generation of future load predictions using trained models.
*   **Customizable Visualizations**: Ability to select date ranges, rolling average windows, and compare historical periods.
*   **Responsive Design**: Utilizes Plotly for interactive and visually appealing charts.

## 🛠️ Tech Stack

*   **Languages**: Python
*   **Frameworks**: Streamlit, Pandas, NumPy, Plotly, Scikit-learn, XGBoost
*   **Data Handling**: CSV processing, DataFrames
*   **Visualization**: Plotly Express, Plotly Graph Objects
*   **Machine Learning**: Regression, Classification, Time Series Analysis

## 📦 Installation

1.  **Clone the repository**: 
    ```bash
    git clone https://github.com/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children.git
    cd System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children
    ```
2.  **Install dependencies**: 
    Ensure you have Python installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Run the Streamlit Application**: 
    Navigate to the project directory in your terminal and run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2.  **Interact with the Dashboard**: 
    The application will open in your web browser. You can:
    *   Upload your own CSV data file (optional) to analyze custom datasets.
    *   Use the sidebar controls to filter data by date range.
    *   Explore different sections and visualizations by navigating through the tabs.
    *   Adjust parameters in the controls to see how they affect the visualizations and predictions.

### Real-world Use Cases

This dashboard can be used by:

*   **Government Agencies (HHS, CBP)**: To monitor current system load, identify potential bottlenecks, and forecast future needs for resource allocation and planning.
*   **Non-profit Organizations**: To understand the scale of operations and advocate for necessary resources and policy changes.
*   **Researchers and Analysts**: To study trends and patterns in the UAC program and contribute to evidence-based policymaking.
*   **Social Workers and Case Managers**: To gain a data-driven understanding of the challenges and demands faced by the system.

## 📁 Project Structure

```
System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children/
├── static/
│   ├── Banner.png
│   ├── Banner_Symbol.png
│   ├── CBP.png
│   └── HHS.png
├── app.py
├── model.py
├── requirements.txt
├── README.md
├── HHS_Unaccompanied_Alien_Children_Program.csv
└── LICENSE
```

## 📝 Table of Contents

*   [Features](#-features)
*   [Tech Stack](#-tech-stack)
*   [Installation](#-installation)
*   [Usage](#-usage)
    *   [Real-world Use Cases](#real-world-use-cases)
*   [Project Structure](#-project-structure)
*   [Table of Contents](#-table-of-contents)
*   [Structural Forecast](#-structural-forecast--system-insights)
    *   [Data Ingestion & Validation](#data-ingestion--validation)
    *   [Individual Trends](#individual-trends-daily-metrics)
    *   [Derived Healthcare Metrics](#derived-healthcare-capacity-metrics)
    *   [Trend & Temporal Analysis](#trend--temporal-analysis)
    *   [Pressure & Stress Identification](#pressure--stress-identification-separate-tab)
*   [Recommendational Forecast](#-recommendational-forecast--feature-engineering-modeling--recommendations)
    *   [Feature Engineering](#feature-engineering)
    *   [Modeling & Evaluation](#modeling--evaluation)
    *   [Alert & Future Predictions](#alert--future-predictions)
*   [Contributing](#contributing)
*   [License](#license)
*   [Important links](#important-links)
*   [Footer](#footer)

---

## 🏗️ Structural Forecast — System Insights

This section provides a **comprehensive diagnostic view** of system capacity and care load. It consolidates notebook visualizations into structured subtabs, ensuring clarity and reduced duplication.

🔹 **Purpose**: Highlight descriptive trends, stress indicators, and healthcare metrics.
🔹 **Format**: Subtabs group related visualizations, while expanders provide concise explanations.
🔹 **Interactivity**: Toggle buttons, plots and hover effects enhance engagement without overwhelming the user.

Navigate through subtabs to explore ingestion checks, individual trends, derived healthcare KPIs, temporal analyses, and stress indicators.

### 📊 Top Key Performance Indicators (KPIs)

*   **👶 Total Children Under Care**: The latest total number of children in the system.
*   **📊 Avg System-wide Responsibility**: The average number of children under care across the entire period.
*   **➕ Positive Net Intake Days**: The count of days where more children were transferred out than discharged.
*   **📈 90th Percentile Load**: The system load that 90% of the days fall below.
*   **🔄 Discharge Offset Ratio**: The ratio of discharged children to transferred children, indicating outflow efficiency.
*   **⚡ Net Intake Pressure**: The latest daily net change in children under care.
*   **📉 Care Load Volatility Index**: A measure of how much the care load fluctuates.
*   **⏳ Backlog Accumulation Rate**: The average positive net intake over the last 7 days, indicating potential backlog buildup.

### 📥 Data Ingestion & ✅ Validation

*   **👀 Demo Dataset Preview**: Displays the first few rows of the loaded dataset, showcasing the data structure after initial processing.
*   **📅 Total Days**: Total number of days covered in the dataset.
*   **⚠️ Missing Dates**: Indicates if there are any gaps in the daily date sequence.
*   **🔁 Duplicate Dates**: Shows if any dates appear more than once, which are aggregated.
*   **🚨 Reporting Anomalies**: Checks for inconsistencies such as transfers exceeding custody numbers or discharges exceeding care numbers, providing samples for review.

### 📈 Individual Trends: Daily Metrics

*   **🌐 3D Consolidated Flow (Metric Lanes)**: A 3D visualization tracking the flow of children across different stages (CBP custody, HHS care, etc.) over time.
*   **📈 Daily Trend**: Interactive line charts to view the daily progression of specific metrics like 'Children in CBP custody'.
*   **🖼️ Snapshot of All Metrics**: Small multiples displaying the daily trend for each key metric for quick comparison.

### ⚕️ Derived Healthcare Metrics

*   **📈 Daily Total System Load**: Tracks the cumulative number of children in CBP and HHS care daily.
*   **📈 Daily Net Daily Intake**: Visualizes the daily net change in children under care, highlighting inflow vs. outflow.
*   **📈 Daily Care Load Growth Rate (%)**: Shows the daily percentage change in total system load, indicating growth or decline.
*   **📊 Backlog Indicator**: Highlights periods of sustained positive net intake, suggesting potential backlog accumulation.
*   **📊 KPI Visualizations**: Interactive 3D plots (Surface Map or Trend-line) showing relationships between Net Daily Intake, Care Load Growth Rate, and Total System Load.
*   **🔄 Dynamic Discharge Cloud**: A 3D scatter plot visualizing intake, discharge ratio, and system load over time.
*   **🗺️ Discharge Load Terrain**: A 3D surface plot illustrating how intake and discharge ratios influence system load.

### ⏳ Trend & Temporal Analysis

*   **📈 Daily Total System Load (CBP +/vs HHS)**: Compares the daily load in CBP custody versus HHS care.
*   **📊 Weekly & Monthly Aggregated Trends**: Shows aggregated system load over weeks and months to identify broader patterns.
*   **⚠️ High‑Load Threshold Analysis**: Identifies and highlights periods where system load exceeds the 90th percentile threshold.
*   **📊 Timeline Comparison**: Allows users to compare total system load across different user-selected periods.
*   **📊 Monthly Total System Load Heatmap**: Visualizes monthly aggregated load across years to spot seasonal patterns.
*   **⚖️ Discharge Effectiveness Over Time**: Tracks the discharge offset ratio to assess outflow efficiency.
*   **🔄 Lag Analysis & Feature Statistics**: Correlation matrix showing the relationship between current load and previous days' loads (lags).
*   **🔀 Flow Efficiency**: Scatter plot examining the relationship between Net Intake and Discharge Offset Ratio.
*   **📈 3D Trend Line**: Visualizes Total System Load, its rolling average, and Net Daily Intake in a 3D space.
*   **📅 3D Temporal Surface**: Surface plot showing system load based on Day of Week and Month.

### ⚠️ Pressure & Stress Identification

*   **📈 Total System Load with Rolling Averages**: Displays daily load alongside customizable rolling averages (7-day, 14-day, and user-selected) to smooth out fluctuations and identify trends.
*   **Detected Strain Windows**: Highlights periods on the Total System Load graph where sustained positive net intake coincides with above-average system load, indicating potential operational strain.
*   **⚠️ 3D Heatmap Surface Plot of System Pressure**: A 3D visualization mapping composite system pressure (derived from Net Daily Intake, Care Load Growth Rate, Sustained Intake, and High Load indicators) against key drivers.

---

## 🎯 Recommendational Forecast — Feature Engineering, Modeling & Recommendations

This section focuses on **predictive modeling and actionable insights**. It builds upon the structural analysis by applying advanced techniques to forecast outcomes and recommend strategies.

*   **📐 Feature Engineering**: Creates lag variables, rolling statistics, and domain-specific features to improve model accuracy.
*   **🤖 Modeling**: Trains regression and classification models to evaluate performance under different scenarios.
*   **📊 Evaluation**: Compares baseline vs engineered features, visualizes accuracy, and assesses volatility.
*   **🔮 Forecasts & Recommendations**: Generates forward-looking predictions and provides decision-ready recommendations.

### 🧩 Feature Engineering

*   **Predictive Power Decay**: Bar chart illustrating the correlation of lagged 'Total System Load' values with the current load, indicating how far back in time features are predictive.
*   **Engineered Features**: Includes lag features (e.g., `lag_1d`), rolling statistics (e.g., `roll_mean_7d`), and cyclical encodings (e.g., `Day_of_Week_sin`, `Month_cos`) to capture temporal patterns.
*   **Feature Interaction**: 3D scatter plot visualizing the relationship between rolling averages (mean and standard deviation) and the target 'Total System Load'.
*   **Date-based Features**: Box plots showing the distribution of 'Total System Load' by Day of Week and Month, revealing seasonal patterns and potential weekly cycles.

### 🧪 Modeling & Evaluation

*   **📈 Regression Models**: Compares performance metrics (MAE, RMSE) of various regression models (Linear Regression, XGBoost, Random Forest, etc.) through individual plots and a comparative heatmap.
*   **🧮 Classification Models**: Evaluates classifiers (Logistic Regression, Random Forest, SVC, etc.) for predicting load categories (Low, Medium, High) using performance metrics (Accuracy, F1-Score, ROC AUC) presented in a heatmap and 3D visualizations.

### 🚨 Alert & 🔮 Future Predictions

*   **📊 Operational Pressure Index (OPI)**: Tracks an index measuring intake velocity against discharge volatility, with alerts triggered when the OPI exceeds a defined threshold.
*   **🌐 3D Pressure Surface**: Visualizes the OPI in relation to system load and variability, highlighting zones of high operational pressure.
*   **Pressure & Stress Insights**: Key metrics for Current OPI, High Pressure Alerts, and Recent Inflow Velocity.
*   **🔮 Future Forecasting**: Uses a retrained Gradient Boosting Regressor model to predict 'Total System Load' for a user-defined future horizon (in months), visualizing historical data alongside the forecast.

## 🌱 Contributing

Contributions are welcome! Please feel free to:

*   Fork the repository.
*   Create a new branch for your feature (`git checkout -b feature/YourFeature`).
*   Commit your changes (`git commit -m 'Add some YourFeature'`).
*   Push to the branch (`git push origin feature/YourFeature`).
*   Open a Pull Request.
*   Please ensure your code adheres to the project's coding standards and includes tests where applicable.

## 📄 License

This project is licensed under the **MIT License**.

[![Forks](https://img.shields.io/github/forks/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children?label=Fork&logo=github)](https://github.com/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children/fork)
[![Stars](https://img.shields.io/github/stars/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children?label=Star&logo=github)](https://github.com/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children/stargazers)
[![Issues](https://img.shields.io/github/issues/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children?label=Open%20Issues&logo=github)](https://github.com/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children/issues)
