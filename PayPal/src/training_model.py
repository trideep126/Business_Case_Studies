import joblib 
import shap
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, roc_auc_score , precision_recall_curve, auc

def train_model(X,y):
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear'
    )
    model.fit(X,y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba =  model.predict_proba(X_test)[:,1]
    precision, recall, _  = precision_recall_curve(y_test,y_pred_proba)
    pr_auc = auc(recall,precision)
    print(f"Classification Report: \n{classification_report(y_test,y_pred)}")
    print(f"Test AUC: {roc_auc_score(y_test,y_pred_proba)}")
    print(f"Test PR AUC: {pr_auc}")

def shap_analyis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values,X_test)
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
    
def save_model(model, path='fraud_detection_model.pkl'):
    #Save model to path
    joblib.dump(model,path)

def load_model(path='fraud_detection_model.pkl'):
    #Load model from path
    return joblib.load(path)