# System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children

### Background and Context:
The Unaccompanied Alien Children (UAC) Program is a federally mandated initiative under which children apprehended by U.S. Customs and Border Protection (CBP) are transferred to the Department of Health and Human Services (HHS) for medical screening, quarantine, sheltering, and eventual placement with vetted sponsors

From a healthcare systems perspective, this program functions as a dynamic care pipeline, involving:

• Intake into CBP custody
• Transfer to HHS care facilities
• Medical, psychological, and welfare support
• Discharge and reunification with sponsors

Effective functioning of this pipeline depends on accurate capacity awareness and care load monitoring, especially during sudden influxes.

**Unified Mentor (https://unifiedmentor.com/)**
**U.S Department of Health and Human Services (https://www.hhs.gov/)**

### Problem Statement
Although daily operational data is collected, HHS currently lacks a centralized analytical framework to continuously assess:

• Total care system load
• Balance between inflow and outflow
• Capacity stress and relief periods
• Sustainability of care delivery over time

Without structured analytics, decision-making becomes reactive, increasing the risk of overcrowding, delayed care, and strain on healthcare infrastructure.

### Primary Objectives
• Quantify daily and cumulative care load across CBP and HHS
• Identify periods of capacity strain and relief
• Analyze balance between intake, transfers, and discharges

### Secondary Objectives
• Support healthcare staffing and shelter planning
• Improve situational awareness for policymakers
• Enable data-driven humanitarian response evaluation

### Dataset Description
Column	Description
Date	Reporting date
Children apprehended and placed in CBP custody	Daily intake volume
Children in CBP custody	Active CBP care load
Children transferred out of CBP custody	Flow into HHS system
Children in HHS Care	Active HHS care load
Children discharged from HHS Care	Successful sponsor placements

### Forecasting Methodology (Step-by-Step)
**Data Ingestion & Structuring**
• Load daily time-series data (2023–2025)
• Convert Date to datetime format
• Ensure chronological ordering
• Create a complete daily index

**Data Quality & Validation**
Derived predictive features include:

• Identify missing or duplicated dates
• Validate logical constraints:
○ Transfers ≤ CBP custody
○ Discharges ≤ HHS care
• Flag reporting anomalies for transparency

**Derived Healthcare Capacity Metrics**
Key computed metrics include:

• Total System Load:
CBP Custody + HHS Care
• Net Daily Intake:
Transfers into HHS − Discharges from HHS
• Care Load Growth Rate
Day-over-day percentage change
• Backlog Indicator
Sustained positive net intake over time

**Trend & Temporal Analysis**
• Daily, weekly, and monthly care load trends
• Identification of sustained high-load periods
• Comparison across calendar periods (early vs late timeline)

**Pressure & Stress Identification**
• Rolling averages (7-day, 14-day)
• Variability analysis
• Detection of prolonged strain windows

### Key Performance Indicators (KPIs)
KPI Name	Description
Total Children Under Care	System-wide responsibility
Net Intake Pressure	Inflow vs outflow imbalance
Care Load Volatility Index	Stability of system
Backlog Accumulation Rate	Sustained care pressure
Discharge Offset Ratio	Ability to relieve load

### Streamlit Web Application Requirements
**Core Modules**
• System Load Overview Pane
• CBP vs HHS Load Comparison
• Net Intake & Backlog Trends
• KPI Summary Cards
**User Capabilities**
• Date range selector
• Metric toggles
• Time granularity filters

### **Deliverables and Submission**
• Research paper (EDA, insights, recommendations)
• Streamlit dashboard (live analytics)
• Executive summary for government stakeholders

## Conclusion
This project delivers a robust, policy-aligned healthcare analytics framework for monitoring the UAC care system. By translating raw counts into structured capacity insights, it empowers stakeholders to understand, evaluate, and improve the delivery of federally mandated child care services.