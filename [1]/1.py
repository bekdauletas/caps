import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        print(f"Successfully loaded data with {len(data)} records")
        print("First few rows:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return pd.DataFrame()


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['Case', 'Stratum', 'Cluster', 'Variable']
    for column in required_columns:
        if column not in data.columns:
            print(f"Error: '{column}' column not found in the dataset.")
            return pd.DataFrame()

    data['response'] = data['Variable']
    data['cluster_id'] = data['Cluster']
    return data


def simple_random_sampling(data: pd.DataFrame) -> Dict[str, float]:
    print("\n==== Simple Random Sampling Analysis ====")
    srs_mean = data['response'].mean()
    print(f"1) Mean (SRS): {srs_mean:.2f}")

    n = len(data)
    srs_std = data['response'].std(ddof=1)
    srs_se = srs_std / np.sqrt(n)
    print(f"2) Standard Error (SRS): {srs_se:.4f}")

    t_value = 2.04
    margin_of_error = t_value * srs_se
    ci_upper = srs_mean + margin_of_error
    ci_lower = srs_mean - margin_of_error
    print(f"3) 95% Confidence Interval (SRS):")
    print(f"   Upper limit: {ci_upper:.4f}")
    print(f"   Lower limit: {ci_lower:.4f}")

    return {
        'srs_mean': srs_mean,
        'srs_se': srs_se,
        'ci_upper': ci_upper,
        'ci_lower': ci_lower
    }


def clustered_random_sampling(data: pd.DataFrame, srs_se: float) -> Dict[str, float]:
    print("\n==== Clustered Random Sampling Analysis ====")
    cluster_means = data.groupby('cluster_id')['response'].mean()
    crs_mean = cluster_means.mean()
    print(f"1) Mean (Clustered): {crs_mean:.2f}")

    M = len(cluster_means)
    cluster_var = np.var(cluster_means, ddof=1)
    crs_se = np.sqrt(cluster_var / M)
    print(f"2) Standard Error (Clustered): {crs_se:.4f}")

    d_value = crs_se / srs_se
    print(f"3) Design Effect (d-value): {d_value:.4f}")

    d_squared = d_value ** 2
    print(f"4) d-squared: {d_squared:.4f}")

    n_avg = data.groupby('cluster_id').size().mean()
    roh = (d_squared - 1) / (n_avg - 1) if n_avg > 1 else 0
    print(f"5) Intraclass correlation (roh): {roh:.4f}")

    Neff = len(data) / d_squared
    print(f"6) Effective sample size (Neff): {Neff:.4f}")

    return {
        'crs_mean': crs_mean,
        'crs_se': crs_se,
        'd_value': d_value,
        'd_squared': d_squared,
        'roh': roh,
        'Neff': Neff
    }


def plot_comparison(srs_mean: float, crs_mean: float, srs_se: float, crs_se: float):
    plt.figure(figsize=(10, 6))
    plt.bar(['SRS Mean', 'Clustered Mean'], [srs_mean, crs_mean])
    plt.errorbar(['SRS Mean', 'Clustered Mean'], [srs_mean, crs_mean],
                 yerr=[srs_se, crs_se], fmt='o', color='black')
    plt.title('Comparison of SRS and Clustered Random Sampling')
    plt.ylabel('Mean Value')
    plt.savefig('sampling_comparison.png')


def analyze_survey_data(excel_file_path: str) -> Dict[str, Any]:
    data = load_data(excel_file_path)
    if data.empty:
        return {}

    data = prepare_data(data)
    if data.empty:
        return {}

    srs_results = simple_random_sampling(data)
    crs_results = clustered_random_sampling(data, srs_results['srs_se'])

    plot_comparison(srs_results['srs_mean'], crs_results['crs_mean'],
                    srs_results['srs_se'], crs_results['crs_se'])

    results = {**srs_results, **crs_results}
    return {k: round(v, 4) for k, v in results.items()}


if __name__ == "__main__":
    results = analyze_survey_data('Question1_Final_CP.xlsx')
    if results:
        print("\n==== Summary of Results (Rounded as Required) ====")
        for key, value in results.items():
            print(f"{key}: {value}")
#srs_mean: 50.6
#srs_se: 6.8851
#ci_upper: 64.6455
#ci_lower: 36.5545
#crs_mean: 50.6
#crs_se: 7.6243
#d_value: 1.1074
#d_squared: 1.2263
#roh: 0.2263
#Neff: 13.0477