import joblib 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, roc_auc_score 

def train_model(X,y):
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    model.fit(X,y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba =  model.predict_proba(X_test)[:,1]
    print(f"Classification Report: \n{classification_report(y_test,y_pred)}")
    print(f"Test AUC: {roc_auc_score(y_test,y_pred_proba)}")

def save_model(model, path='fraud_detection_model.pkl'):
    #Save model to path
    joblib.dump(model,path)

def load_model(path='fraud_detection_model.pkl'):
    #Load model from path
    return joblib.load(path)