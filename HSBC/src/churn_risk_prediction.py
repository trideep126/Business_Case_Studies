import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , roc_auc_score 



def load_clustered_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0',axis=1,inplace=True)
    return df 

def create_churn_risk_features(data: pd.DataFrame):

    df = data.copy()

        # Risk indicators based on banking domain knowledge
    df['low_balance_risk'] = (df['CustAccountBalance'] < df['CustAccountBalance'].quantile(0.25)).astype(int)
    df['high_transaction_ratio_risk'] = (df['BalTransRatio'] > df['BalTransRatio'].quantile(0.75)).astype(int)
    df['age_risk'] = ((df['Age'] < 25) | (df['Age'] > 65)).astype(int)
    df['infrequent_user_risk'] = (df['TransactionAmount'] < df['TransactionAmount'].quantile(0.25)).astype(int)

    # Create synthetic churn labels based on risk factors (for demonstration)
    # In real scenario, this would come from historical churn data
    df['churn_risk_score'] = (
        df['low_balance_risk'] * 0.3 +
        df['high_transaction_ratio_risk'] * 0.25 +
        df['age_risk'] * 0.2 +
        df['infrequent_user_risk'] * 0.25
        )

    # Binary churn risk (high risk = 1, low risk = 0)
    df['high_churn_risk'] = (df['churn_risk_score'] > 0.5).astype(int)

    return df

def train_churn_model(data: pd.DataFrame):
    scaler= StandardScaler()

    df = create_churn_risk_features(data)

    # Features for churn prediction
    feature_cols = ['CustAccountBalance', 'TransactionAmount', 'Age', 'BalTransRatio',
                       'CustGender', 'TransactionMonth', 'Cluster']

    X = df[feature_cols]
    y = df['high_churn_risk']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Model performance
    print("\n=== CHURN RISK MODEL PERFORMANCE ===")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop Features for Churn Prediction:")
    print(feature_importance.head())

    return df

def segment_churn_analysis(df):

    churn_by_segment = df.groupby('Cluster').agg({
        'high_churn_risk': ['count', 'sum', 'mean'],
        'churn_risk_score': 'mean'
    }).round(3)

    churn_by_segment.columns = ['total_customers', 'high_risk_customers', 'churn_rate', 'avg_risk_score']

    return churn_by_segment