import pandas as pd

def design_segment_experiments():

    experiments = {
        'Segment_0': {
            'experiment_name': 'Mobile Banking Adoption',
            'hypothesis': 'Personalized mobile app tutorials increase transaction frequency',
            'control_group_size': 0.5,
            'test_duration_days': 60,
            'primary_metric': 'transaction_frequency',
            'success_criteria': 'increase_by_15_percent',
            'estimated_impact': {
                'revenue_uplift': 180,  # per customer annually
                'confidence_level': 0.95
            }
        },
        'Segment_1': {
            'experiment_name': 'Premium Product Cross-sell',
            'hypothesis': 'Targeted investment product offers increase portfolio adoption',
            'control_group_size': 0.5,
            'test_duration_days': 90,
            'primary_metric': 'product_adoption_rate',
            'success_criteria': 'increase_by_25_percent',
            'estimated_impact': {
                'revenue_uplift': 1200,  # per customer annually
                'confidence_level': 0.95
            }
        },
        'Segment_2': {
            'experiment_name': 'Credit Product Optimization',
            'hypothesis': 'Simplified credit application process increases approval rates',
            'control_group_size': 0.5,
            'test_duration_days': 45,
            'primary_metric': 'credit_application_rate',
            'success_criteria': 'increase_by_20_percent',
            'estimated_impact': {
                'revenue_uplift': 800,  # per customer annually
                'confidence_level': 0.95
            }
        }
    }

    return experiments


def calculate_test_requirements(data:pd.DataFrame, experiments):

    segment_counts = data['Cluster'].value_counts()

    test_design = {}

    for segment_name, exp_details in experiments.items():
        segment_id = int(segment_name.split('_')[1])
        total_customers = segment_counts[segment_id]

        # Sample size calculation (simplified)
        control_size = int(total_customers * exp_details['control_group_size'])
        test_size = total_customers - control_size

        test_design[segment_name] = {
            'total_eligible_customers': total_customers,
            'control_group_size': control_size,
            'test_group_size': test_size,
            'test_duration': exp_details['test_duration_days'],
            'expected_revenue_impact': exp_details['estimated_impact']['revenue_uplift'] * test_size,
            'experiment_name': exp_details['experiment_name']
        }

    return test_design