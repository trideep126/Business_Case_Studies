import joblib
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metric import classification_report, roc_auc_score

def train_model(X,y):
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth = 6,
        random_state=42,
        verbose=0
    )
    model.fit(X,y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    print(f'Classification Report: \n{classification_report(y_test, y_pred)}')
    print(f'Test AUC: {roc_auc_score(y_test,y_pred_proba)}')

def shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test.iloc[:1000], plot_type="bar",show=False)
    plt.title("SHAP Feature Importance - Credit Risk Drivers")
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png',dpi=300, bbox_inches='tight')

def save_model(model, path='credit_risk_model.pkl'):
    #Save model to file using joblib
    joblib.dump(model,path)

def load_model(path='credit_risk_model.pkl'):
    #Load model from file using joblib
    return joblib.load(path)



