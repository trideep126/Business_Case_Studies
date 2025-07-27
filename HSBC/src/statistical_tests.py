import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import  ols 

#Perform t-test between customer gender and account balance
def perform_t_test1(df: pd.DataFrame):
    male_customers = df[df['CustGender'] == 0]['CustAccountBalance']
    female_customers = df[df['CustGender'] == 1]['CustAccountBalance']

    t_stat,p_value = stats.ttest_ind(male_customers,female_customers)

    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    alpha= 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in average Customer Account Balance between males and females")
    else:
        print("Fail to reject the null hypothesis: There's no significant difference in average Customer Account Balance")


#Perform t-test between customer age and transaction amount
def perform_t_test2(df: pd.DataFrame):
    male_customers = df[df['CustGender'] == 0]['TransactionAmount']
    female_customers = df[df['CustGender'] == 1]['TransactionAmount']

    t_stat, p_value = stats.ttest_ind(male_customers,female_customers)

    print(f"T-Statistic: {t_stat}, P-value: {p_value}")

    alpha=0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There's a significant difference between average transaction amount between males and females")
    else:
        print("Fail to reject the null nypothesis: Theres' no significant difference in avg trasnaction amount across males and females")

#Perform Anova
def perform_anova1(df: pd.DataFrame):
    anova_data = df[['CustAccountBalance','Age']]

    model = ols('CustAccountBalance ~ C(Age)',data = anova_data).fit()
    anova_table = sm.stats.anova_lm(model,typ=3)

    alpha = 0.05
    p_value = anova_table['PR(>F)'][1]

    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in avg customer account balance across Age groups ")
    else:
        print("Fail to reject null hypothesis: There's no significant difference in avg customer acount balance across Age groups")


def perform_anova2(df: pd.DataFrame):
    anova_data = df[['TransactionAmount','Age']]

    model = ols('TransactionAmount ~ C(Age)',data = anova_data).fit()
    anova_table = sm.stats.anova_lm(model,typ=3)

    alpha = 0.05
    p_value = anova_table['PR(>F)'][1]

    if p_value < alpha:
        print("Reject the null hypothesis: There's a significance difference in avg TransactionAMount across age groups")
    else:
        print("Fail to reject the null  hypothesis: There's no significant difference in avg Transaction Amount across age groups")

