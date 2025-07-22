import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder 

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert features into lower case
    df.columns = df.columns.str.lower()

    # drop unnecessary columns
    df.drop(['id','year'],axis=1,inplace=True)

    # rename columns for better understanding
    df.rename(columns={
        'co-applicant_credit_type': 'co_applicant_credit_type',
        'ltv':'lifetime_value',
        'dtir1':'debt_to_income_ratio'
    },inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # separate numerical and categorical columns
    num_columns = df.select_dtypes(include=[np.number]).columns
    cat_columns = df.select_dtypes(include=[object]).columns

    num_columns = num_columns.drop('status',errors='ignore') # Exclude target variable from numerical columns

    # fill numerical features with median
    for col in num_columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(),inplace=True)

    # fill categorical features with mode
    for col in cat_columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0],inplace=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    #Engineer new features
    df['loan_to_value_ratio'] = df['loan_amount'] / df['property_value']
    df['interest_burden'] = df['rate_of_interest'] / df['loan_amount'] * 100
    df['risk_adjusted_ltv'] = df['lifetime_value'] / (1 + df['debt_to_income_ratio'])

def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    #Select relevant features
    selected_features = ['income','credit_score','debt_to_income_ratio','loan_amount','rate_of_interest','term','age',
                'property_value','secured_by','lifetime_value','loan_type','interest_only','neg_ammortization',
                'lump_sum_payment','open_credit','credit_type','co_applicant_credit_type','loan_to_value_ratio',
                'interest_burden','risk_adjusted_ltv']
    return df[selected_features]

def preprocess_features(X):
    label_encoder = {}
    scaler = StandardScaler()

    X_processed = X.copy()
    #Encode categorical variables
    cat_columns = X_processed.select_dtypes(include=[object]).columns
    for col in cat_columns:
        if col not in label_encoder:
            label_encoder[col] = LabelEncoder()
            X_processed[col] = label_encoder[col].fit_transform(X_processed[col])
        else:
            X_processed[col] = label_encoder[col].transform(X_processed[col])

    #Scale numerical variables
    num_columns = X_processed.select_dtypes(include=[np.number]).columns
    X_processed[num_columns] = scaler.fit_transform(X_processed[num_columns])

    return X_processed