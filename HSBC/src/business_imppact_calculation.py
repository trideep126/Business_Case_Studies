def calculate_total_business_impact(data,clv_metrics,product_recommendations, test_design):

    segment_counts = data['Cluster'].value_counts()

    # Current business value
    current_value = 0
    for segment in range(3):
        segment_size = segment_counts[segment]
        clv_per_customer = clv_metrics.loc[segment, 'customer_lifetime_value']
        current_value += segment_size * clv_per_customer

    # Potential value from product recommendations
    product_value = 0
    for segment_name, recs in product_recommendations.items():
        top_product = recs[0]  # Best ROI product
        product_value += top_product['projected_revenue']

    # A/B testing potential uplift
    ab_testing_value = sum(design['expected_revenue_impact'] for design in test_design.values())

    # Churn prevention value (assume 5% churn rate reduction saves 10% of CLV)
    churn_prevention_value = current_value * 0.05 * 0.10

    impact_summary = {
        'current_customer_base_value': current_value,
        'product_cross_sell_opportunity': product_value,
        'ab_testing_uplift_potential': ab_testing_value,
        'churn_prevention_value': churn_prevention_value,
        'total_additional_value_potential': product_value + ab_testing_value + churn_prevention_value,
        'roi_multiple': (product_value + ab_testing_value + churn_prevention_value) / current_value
    }

    return impact_summary