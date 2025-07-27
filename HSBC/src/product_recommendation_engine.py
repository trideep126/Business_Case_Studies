import pandas as pd

def define_product_portfolio():

    products = {
        'Credit_Card': {
            'segment_affinity': {0: 0.3, 1: 0.8, 2: 0.6},
            'revenue_per_customer': 1200,
            'acquisition_cost': 150
        },
        'Personal_Loan': {
            'segment_affinity': {0: 0.2, 1: 0.6, 2: 0.7},
            'revenue_per_customer': 2400,
            'acquisition_cost': 300
        },
        'Investment_Portfolio': {
            'segment_affinity': {0: 0.1, 1: 0.9, 2: 0.4},
            'revenue_per_customer': 3600,
            'acquisition_cost': 500
        },
        'Premium_Banking': {
            'segment_affinity': {0: 0.1, 1: 0.8, 2: 0.3},
            'revenue_per_customer': 4800,
            'acquisition_cost': 800
        },
        'Mobile_Banking_Plus': {
            'segment_affinity': {0: 0.7, 1: 0.5, 2: 0.8},
            'revenue_per_customer': 360,
            'acquisition_cost': 50
        }
    }

    return products


def calculate_product_recommendations(data: pd.DataFrame):

    products = define_product_portfolio()
    segment_counts = data['Cluster'].value_counts()

    recommendations = {}

    for segment in sorted(data['Cluster'].unique()):
        segment_size = segment_counts[segment]
        segment_recs = []

        for product, details in products.items():
            affinity = details['segment_affinity'][segment]
            expected_customers = segment_size * affinity

            revenue = expected_customers * details['revenue_per_customer']
            cost = expected_customers * details['acquisition_cost']
            roi = (revenue - cost) / cost if cost > 0 else 0

            segment_recs.append({
                'product': product,
                'affinity_score': affinity,
                'expected_adopters': int(expected_customers),
                'projected_revenue': revenue,
                'acquisition_cost': cost,
                'roi': roi
            })

        # Sort by ROI
        segment_recs.sort(key=lambda x: x['roi'], reverse=True)
        recommendations[f'Segment_{segment}'] = segment_recs

    return recommendations