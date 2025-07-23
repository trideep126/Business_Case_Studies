import shap
import pandas as pd
def production_score_application(application_data,model,threshold=None,X_train=None,optimal_threshold=None):
    if threshold is None:
        threshold = optimal_threshold

    risk_score = model.predict_proba(application_data.reshape(1,-1))[0,1]

    if risk_score >= threshold:
        decision = "DECLINE"
        reason = "High Default Risk"
    else:
        decision = "APPROVE"
        reason = "Acceptable Risk Profile"

    shap_values_individual = explainer.shap_values(application_data.reshape(1,-1))[0]
    top_risk_factors = []

    for i,feature in enumerate(X_train.columns):
        if abs(shap_values_individual[i]) > 0.01:
            top_risk_factors.append(f"{feature}: {shap_values_individual[i]:.3f}")

    return {
        'risk_score': risk_score,
        'decision': decision,
        'reason': reason,
        'top_risk_factors': top_risk_factors[:3], #top 3
        'model_version': 'CatBoos v1.0',
        'timestamp': pd.Timestamp.now().isoformat()
    }
