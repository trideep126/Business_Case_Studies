import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    #rename columns for better understanding
    df.rename(columns={
        'TransactionAmount (INR)': 'TransactionAmount'
    }, inplace=True)

    #convert date columns to proper datetime format
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'])
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

    #calculate Age from CustoemrDOB
    today = datetime.today().year 
    df['Age'] = today - df['CustomerDOB'].dt.year
    df['Age'] = df['Age'].astype(int)

    #only keep customer with valid age
    df = df[df['Age'].between(18,90)]

    #drop any null values
    df.dropna(inplace=True)

    #Drop duplicates
    df.drop_duplicates(subset='CustomerID',inplace=True)

    #keep only popular customer locations
    value_counts = df['CustLocation'].value_counts()
    value_counts_df = value_counts.reset_index()
    value_counts_df.columns = ['CustLocation','ValueCounts']
    locations = value_counts_df[value_counts_df['ValueCounts'].between(1000,75992)]
    locations_arr = list(locations['CustLocation'])
    df = df[df['CustLocation'].isin(locations_arr)]

    #add new feature Balance-Transaction Ratio
    df['BalTransRatio'] = df['TransactionAmount'] / df['CustAccountBalance']

    #replace infinite values
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(subset=['CustAccountBalance','TransactionAmount','Age','BalTransRatio'],inplace=True)

    #remove those transactions where transaction amount is greater than customer account balance
    df = df[~(df['TransactionAmount'] > df['CustAccountBalance'])]

    #Drop transaction time 
    df.drop('TransactionTime',axis=1,inplace=True)

    #Winsorization
    df = df[df['CustAccountBalance'].isin(winsorize(df['CustAccountBalance'],limits=(0,0.25)))]
    df = df[df['TransactionAmount'].isin(winsorize(df['TransactionAmount'],limits=(0,0.25)))]

    #need the transaction month
    df['TransactionMonth'] = df['TransactionDate'].dt.month
    df.drop('TransactionDate',axis=1,inplace=True)

    return df 


def Label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    #Label eencoding Genders
    encoder = LabelEncoder()
    df['CustGender'] = encoder.fit_transform(df['CustGender'])

    #Using Dummy variables for Customer Location
    dummies = pd.get_dummies(df['CustLocation'])
    dummies = dummies.astype(int)

    new_df = df.merge(dummies,left_index=True,right_index=True)
    new_df.drop('CustLocation',axis=1,inplace=True)

    return new_df

