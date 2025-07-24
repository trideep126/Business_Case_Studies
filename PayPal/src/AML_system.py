import pandas as pd

def create_monitoring_data(model,X_test,y_test) -> pd.DataFrame:
    risk_scores = model.predict_proba(X_test)[:,1]

    risk_threshold = {'low':0.3, 'medium':0.7, 'high': 0.9}

    df = pd.DataFrame({
        'transaction_id': range(len(risk_scores)),
        'risk_score': risk_scores,
        'actual_fraud': y_test.values,
        'risk_level': pd.cut(risk_scores,
                             bins=[0,risk_threshold['low'],
                             risk_threshold['medium'],
                             risk_threshold['high'],1],
                             labels=['Low','Medium','High','Critical'])
    })
    return df 

def apply_aml_rules(row):
    rules_triggered = []
    risk_threshold = {'low':0.3, 'medium':0.7, 'high': 0.9}

    if row['risk_score'] > risk_threshold['high']:
        rules_triggered.append('HIGH RISK SCORE')
    
    if row['risk_level'] == 'Critical':
        rules_triggered.append('CRITICAL RISK LEVEL')

    return '|'.join(rules_triggered) if rules_triggered else 'NO ALERT'

def generate_monitoring_report(monitoring_data: pd.DataFrame):
    print("\n AML Monitoring report")
    print(f"Total Transaction Monitored: {len(monitoring_data)}")
    print("\n Risk Level Distribution: ")
    print(monitoring_data['risk_level'].value_counts())

    print("\n AML Alerts Summary: ")
    alert_summary = monitoring_data['aml_alerts'].value_counts()
    print(alert_summary)

def monitoring_performance_metrics(monitoring_data: pd.DataFrame):
    risk_threshold = {'low':0.3, 'medium':0.7, 'high': 0.9}
    high_risk_transactions = monitoring_data[monitoring_data['risk_score'] > risk_threshold['high']]

    true_positives = len(monitoring_data[(monitoring_data['risk_score'] > risk_threshold['medium']) & 
                                         (monitoring_data['actual_fraud'] == 1)])
    
    false_positives = len(monitoring_data[(monitoring_data['risk_score'] > risk_threshold['medium']) & 
                                         (monitoring_data['actual_fraud'] == 0)])
    
    precision = true_positives / (true_positives + false_positives)

    print("Monitoring System Performance:")
    print(f"Transactions requiring Manual Review: {high_risk_transactions}")
    print(f"Alert Precision: {precision: .3f}")
    print(f"Alert Detection Rate: {true_positives / monitoring_data['actual_fraud'].sum(): .3f}")