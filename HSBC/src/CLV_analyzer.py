import pandas as pd

def calculate_clv_metrics(clustered_data: pd.DataFrame):

        # Basic segment analysis
        segment_metrics = clustered_data.groupby('Cluster').agg({
            'TransactionAmount': ['mean', 'count', 'std', 'sum'],
            'CustAccountBalance': ['mean', 'std'],
            'Age': ['mean', 'std'],
            'BalTransRatio': ['mean', 'std']
        }).round(2)

        # Flatten column names
        segment_metrics.columns = ['_'.join(col).strip() for col in segment_metrics.columns]

        # Calculate business metrics
        segment_metrics['avg_monthly_transactions'] = segment_metrics['TransactionAmount_count'] / 12
        segment_metrics['annual_transaction_volume'] = segment_metrics['TransactionAmount_sum'] * 12

        # Revenue assumptions (typical banking metrics)
        transaction_fee_rate = 0.015  # 1.5% per transaction
        balance_interest_margin = 0.02  # 2% net interest margin

        # Calculate revenue streams
        segment_metrics['annual_transaction_revenue'] = (
            segment_metrics['annual_transaction_volume'] * transaction_fee_rate
        )

        segment_metrics['annual_balance_revenue'] = (
            segment_metrics['CustAccountBalance_mean'] * balance_interest_margin
        )

        segment_metrics['total_annual_revenue_per_customer'] = (
            segment_metrics['annual_transaction_revenue'] +
            segment_metrics['annual_balance_revenue']
        )

        # Customer lifecycle assumptions
        avg_customer_lifespan = {'0': 3.5, '1': 5.2, '2': 4.1}  # years, based on segment behavior

        segment_metrics['estimated_lifespan_years'] = segment_metrics.index.map(
            lambda x: avg_customer_lifespan[str(x)]
        )

        # Calculate CLV with discount rate (10% annual)
        discount_rate = 0.10
        segment_metrics['customer_lifetime_value'] = (
            segment_metrics['total_annual_revenue_per_customer'] *
            segment_metrics['estimated_lifespan_years'] *
            (1 - discount_rate)
        )

        return segment_metrics

def segment_business_profiles(clustered_data: pd.DataFrame):

        profiles = {}

        # Get customer counts per segment
        segment_counts = clustered_data['Cluster'].value_counts().sort_index()

        for cluster in sorted(clustered_data['Cluster'].unique()):
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster]

            profiles[f'Segment_{cluster}'] = {
                'customer_count': len(cluster_data),
                'percentage_of_base': (len(cluster_data) / len(clustered_data)) * 100,
                'avg_account_balance': cluster_data['CustAccountBalance'].mean(),
                'avg_transaction_amount': cluster_data['TransactionAmount'].mean(),
                'transaction_frequency': cluster_data['TransactionAmount'].count(),
                'gender_split': cluster_data['CustGender'].value_counts(normalize=True).to_dict(),
                'age_range': f"{cluster_data['Age'].min()}-{cluster_data['Age'].max()}",
                'avg_age': cluster_data['Age'].mean(),
                'top_locations': cluster_data['CustLocation'].value_counts().head(3).to_dict()
            }

        return profiles