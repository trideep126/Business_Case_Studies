# PayPal Fraud Detection & AML Transaction Monitoring Case Study

## Executive Summary

Developed an advanced Fraud Detection system achieving **83% Alert Precision Rate** and **$1.2M Prevented losses** per 300K transactions using Machine Learning and real-time transaction monitoring. This case study demonstrates enterprise-grade ML engineering applied to PayPal-style payment fraud detection.

**Key Achievements:**
- 📊 **Fraud Detection with ML models**
- 💰 **Business Impact Calculation** 
- ⚡ **Real-Time Transaction Monitoring**
- 🔄 **Model Drift Monitoring**

## 🎯 Business Impact

| Metric | Value | Business Significance |
|--------|-------|---------------------|
| **Frauds Detected** | 9K | Direct loss prevention |
| **Losses Prevented** | $1.3M | Estimated based on detected frauds |
| **Alert Precision** | 82.7% | Accuracy of high-risk alerts |
| **Detection Rate** | 0.49 | Proportion of actual frauds detected |

## Technical Architecture

### Data Pipeline
- **Dataset**: Nearly 1 million transaction records
- **Features**: 8 original + 5 engineered features
- **Preprocessing**: Feature engineering, Feature Scaling, Handling Class Imbalance
- **Validation**: Data Splitting with stratification

### Model Performance
```
Logistic Regression Results:
├── Accuracy: 95%
├── Precision: 74%
├── Recall: 69%
├── F1-Score: 71%
└── AUC: 0.98
└── PR-AUC: 0.79
```

### Top Fraud Indicators
1. **Median of Purchase Price** (34% importance) - Primary indicator
2. **Online Order** (20% importance)
3. **High Value** (14% importance)
4. **Distance from Home** (13% importance)  
5. **Security Score** (5% importance)

## Project Structure

```
├── notebooks/
│   ├── data_extraction.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_comparison_and_evaluation.ipynb
│   └── AML_transaction_system.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── training_model.py
│   ├── AML_system.py
│   └── business_impact_calculator.py
├── models/
│   ├── fraud_final_model.pkl
├── docs/
│   ├── feature_description.pdf
│   ├── business_workflow.pdf
│   └── dataflow_diagram.pdf
├── reports/
│   ├── model_card.txt
│   ├── business_impact_report.pdf
│   └── executive_dashboard.png
└── README.md
```

## AML Monitoring System

The AML monitoring system utilizes the trained Logistic Regression model to assign risk scores to transactions and categorize them into different risk levels (Low, Medium, High, Critical) based on defined thresholds.

- **Risk Scoring**: Based on the predicted probability of fraud from the Logistic Regression model.
- **Risk Levels**: Defined based on thresholds (e.g., Low < 0.3, Medium 0.3-0.7, High 0.7-0.9, Critical > 0.9).
- **AML Alerts**: Generated based on predefined rules, such as a high risk score or critical risk level.

## Model Drift Monitoring

A function `monitor_model_drift` is included to check for potential drift in feature distributions on new data compared to the original training data. Currently, it checks for a >20% change in the mean of 'distance_from_home' and 'ratio_to_median_purchase_price'.

## Regulatory Compliance

While a full production system would require rigorous adherence to regulatory standards, this case study demonstrates the foundational elements for AML compliance, including:

- **Transaction Monitoring**: Real-time risk scoring and alerting.
- **Risk-Based Approach**: Categorizing transactions by risk level.
- **Suspicious Activity Detection**: Flagging transactions based on rules and risk scores.

Further development would involve integrating with systems for KYC, OFAC screening, and detailed reporting for regulatory bodies.

## Business Applications

### Immediate Use Cases
- **Real-time Transaction Screening** - Instant fraud detection based on risk scores.
- **AML Monitoring** - Identifying transactions requiring further review.

### Strategic Value
- **Risk Reduction** - Proactive identification of potentially fraudulent transactions.
- **Operational Efficiency** - Focusing manual review efforts on high-risk alerts.

## Future Enhancements

Based on the analysis and the current implementation, potential future enhancements include:

### Phase 1: Advanced Detection
- Exploring more complex models or ensemble techniques.
- Incorporating temporal features or sequence analysis.

### Phase 2: AI/ML Improvements
- Implementing continuous model retraining pipelines.
- Exploring more sophisticated model drift detection methods.

### Phase 3: Integration
- Building a production-ready system with real-time data streaming and alerting.
- Integrating with case management systems for alert investigation.

## Technical Stack

The technical stack used in this notebook includes:

- **Data Analysis & Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: sklearn, imblearn, lightgbm, xgboost
- **Model Interpretability**: shap

A production system would likely involve additional technologies for stream processing, feature stores, monitoring, and deployment as outlined in your original sample.

## Key Differentiators

**What makes this solution unique (based on the notebook):**
- ✅ **End-to-End Workflow**: Demonstrates data loading, exploration, feature engineering, modeling, evaluation, and basic AML monitoring.
- ✅ **Model Comparison**: Evaluates multiple machine learning models for fraud detection.
- ✅ **Business Impact**: Provides a basic calculation of potential business impact.
- ✅ **Model Drift Consideration**: Includes a simple function to check for potential model drift.

## About This Project

This case study demonstrates a practical approach to building a fraud detection and AML transaction monitoring system using machine learning, based on a synthetic dataset.

**Skills Demonstrated:**
- Data Analysis and Visualization
- Feature Engineering
- Machine Learning Model Development and Evaluation
- Handling Class Imbalance
- Basic AML Monitoring Concepts
- Model Drift Monitoring
---
