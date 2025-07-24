import pandas as pd

def calculate_business_impact(monitoring_data: pd.DataFrame):
    avg_fraud_loss = 150
    fp_cost = 2

    detected_frauds = len(monitoring_data[(monitoring_data['risk_score'] > 0.7) &
                                          (monitoring_data['actual_fraud'] == 1)])
    
    prevented_losses = detected_frauds * avg_fraud_loss 
    operational_costs = len(monitoring_data[monitoring_data['risk_score'] > 0.7]) * fp_cost

    roi = (prevented_losses - operational_costs) * operational_costs * 100

    print("Business Impact:")
    print(f"Frauds Detected : {detected_frauds}")
    print(f"Losses Prevented: ${prevented_losses:,}")
    print(f"ROI: {roi: .1f}%")


def monitor_model_drift(original_data,new_data):
    # Compare feature distributions
    drift_detected = []

    for feature in ['distance_from_home', 'ratio_to_median_purchase_price']:
        original_mean = original_data[feature].mean()
        new_mean = new_data[feature].mean()

        drift_pct = abs(new_mean - original_mean) / original_mean * 100

        if drift_pct > 20:  # 20% threshold
            drift_detected.append(feature)

    if drift_detected:
        print(f"⚠️ Model Drift Detected in: {drift_detected}")
        print("Recommendation: Retrain model")
    else:
        print("✅ No significant drift detected")