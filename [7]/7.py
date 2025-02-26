import pandas as pd
import numpy as np

# Read the dataset from CSV file
data = pd.read_csv('Question7_Final_CP.csv')

# Define category-specific parameters for Age Group (as provided in your previous request)
category_specific_params_age = {
    "18-24": {"rho": 0.020, "m": 5},
    "25-34": {"rho": 0.025, "m": 6},
    "35-44": {"rho": 0.030, "m": 5},
    "45-54": {"rho": 0.022, "m": 4},
    "55-64": {"rho": 0.018, "m": 5},
    "65+": {"rho": 0.015, "m": 6},
}

# Default clustering parameters for other categories (you can adjust these)
default_params = {"rho": 0.020, "m": 5}  # Default values for Race/Ethnicity, Gender, Income Level

# Categories to analyze
categories = ['Age Group', 'Race/Ethnicity', 'Gender', 'Income Level']

# Initialize results dictionary
results = {}

# Total sample size
total_sample_size = len(data)

# Calculate for each category
for category in categories:
    # Get unique values in the category
    unique_values = data[category].unique()

    for value in unique_values:
        # Filter data for this specific category value
        count = len(data[data[category] == value])

        # 1. Count
        count_value = count

        # 2. Proportion (p)
        proportion = count_value / total_sample_size

        # 3. Standard Error (SE)
        se = np.sqrt((proportion * (1 - proportion)) / total_sample_size)

        # 4. 95% Confidence Interval (CI)
        z = 1.96  # For 95% confidence level
        ci_lower = proportion - z * se
        ci_upper = proportion + z * se

        # 5. Get rho and m for the category
        if category == 'Age Group':
            params = category_specific_params_age.get(value, default_params)
        else:
            params = default_params  # Use default for Race/Ethnicity, Gender, Income Level

        rho = params["rho"]
        m = params["m"]

        # 6. Design Effect (DEFF)
        deff = 1 + rho * (m - 1)

        # 7. Adjusted Standard Error (SE_adj)
        se_adj = se * np.sqrt(deff)

        # 8. Updated 95% Confidence Interval with adjusted SE
        ci_lower_adj = proportion - z * se_adj
        ci_upper_adj = proportion + z * se_adj

        # Store results
        results[(category, value)] = {
            "Count": count_value,
            "Proportion (p)": proportion,
            "Standard Error (SE)": se,
            "95% CI (Unadjusted)": (ci_lower, ci_upper),
            "rho": rho,
            "m": m,
            "Design Effect (DEFF)": deff,
            "Adjusted Standard Error (SE_adj)": se_adj,
            "95% CI (Adjusted)": (ci_lower_adj, ci_upper_adj)
        }

# Print results for each category and value
for (cat, value), metrics in results.items():
    print(f"\n{cat}: {value}")
    print(f"1. Count: {metrics['Count']}")
    print(f"2. Proportion (p): {metrics['Proportion (p)']:.4f}")
    print(f"3. Standard Error (SE): {metrics['Standard Error (SE)']:.4f}")
    print(f"4. 95% CI (Unadjusted): [{metrics['95% CI (Unadjusted)'][0]:.4f}, {metrics['95% CI (Unadjusted)'][1]:.4f}]")
    print(f"5. Intraclass Correlation (rho): {metrics['rho']}")
    print(f"6. Average Cluster Size (m): {metrics['m']}")
    print(f"7. Design Effect (DEFF): {metrics['Design Effect (DEFF)']:.4f}")
    print(f"8. Adjusted Standard Error (SE_adj): {metrics['Adjusted Standard Error (SE_adj)']:.4f}")
    print(f"9. 95% CI (Adjusted): [{metrics['95% CI (Adjusted)'][0]:.4f}, {metrics['95% CI (Adjusted)'][1]:.4f}]")

'''Age Group: 18-24

Count: 227
Proportion (p): 0.1513
Standard Error (SE): 0.0093
95% CI (Unadjusted): [0.1332, 0.1695]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0096
95% CI (Adjusted): [0.1325, 0.1702

Age Group: 55-64

Count: 212
Proportion (p): 0.1413
Standard Error (SE): 0.0090
95% CI (Unadjusted): [0.1237, 0.1590]
Intraclass Correlation (rho): 0.018
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0720
Adjusted Standard Error (SE_adj): 0.0093
95% CI (Adjusted): [0.1231, 0.1596]
'''