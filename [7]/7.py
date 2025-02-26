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

'''Age Group: 65+

Count: 153
Proportion (p): 0.1020
Standard Error (SE): 0.0078
95% CI (Unadjusted): [0.0867, 0.1173]
Intraclass Correlation (rho): 0.015
Average Cluster Size (m): 6
Design Effect (DEFF): 1.0750
Adjusted Standard Error (SE_adj): 0.0081
95% CI (Adjusted): [0.0861, 0.1179]
Age Group: 35-44

Count: 336
Proportion (p): 0.2240
Standard Error (SE): 0.0108
95% CI (Unadjusted): [0.2029, 0.2451]
Intraclass Correlation (rho): 0.03
Average Cluster Size (m): 5
Design Effect (DEFF): 1.1200
Adjusted Standard Error (SE_adj): 0.0114
95% CI (Adjusted): [0.2017, 0.2463]
Age Group: 18-24

Count: 227
Proportion (p): 0.1513
Standard Error (SE): 0.0093
95% CI (Unadjusted): [0.1332, 0.1695]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0096
95% CI (Adjusted): [0.1325, 0.1702]
Age Group: 45-54

Count: 188
Proportion (p): 0.1253
Standard Error (SE): 0.0085
95% CI (Unadjusted): [0.1086, 0.1421]
Intraclass Correlation (rho): 0.022
Average Cluster Size (m): 4
Design Effect (DEFF): 1.0660
Adjusted Standard Error (SE_adj): 0.0088
95% CI (Adjusted): [0.1080, 0.1426]
Age Group: 25-34

Count: 384
Proportion (p): 0.2560
Standard Error (SE): 0.0113
95% CI (Unadjusted): [0.2339, 0.2781]
Intraclass Correlation (rho): 0.025
Average Cluster Size (m): 6
Design Effect (DEFF): 1.1250
Adjusted Standard Error (SE_adj): 0.0120
95% CI (Adjusted): [0.2326, 0.2794]
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
Race/Ethnicity: Mexican

Count: 121
Proportion (p): 0.0807
Standard Error (SE): 0.0070
95% CI (Unadjusted): [0.0669, 0.0944]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0073
95% CI (Adjusted): [0.0663, 0.0950]
Race/Ethnicity: Other Hispanic

Count: 51
Proportion (p): 0.0340
Standard Error (SE): 0.0047
95% CI (Unadjusted): [0.0248, 0.0432]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0049
95% CI (Adjusted): [0.0245, 0.0435]
Race/Ethnicity: White

Count: 1072
Proportion (p): 0.7147
Standard Error (SE): 0.0117
95% CI (Unadjusted): [0.6918, 0.7375]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0121
95% CI (Adjusted): [0.6909, 0.7384]
Race/Ethnicity: Black

Count: 175
Proportion (p): 0.1167
Standard Error (SE): 0.0083
95% CI (Unadjusted): [0.1004, 0.1329]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0086
95% CI (Adjusted): [0.0998, 0.1335]
Race/Ethnicity: Other

Count: 81
Proportion (p): 0.0540
Standard Error (SE): 0.0058
95% CI (Unadjusted): [0.0426, 0.0654]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0061
95% CI (Adjusted): [0.0421, 0.0659]
Gender: Female

Count: 752
Proportion (p): 0.5013
Standard Error (SE): 0.0129
95% CI (Unadjusted): [0.4760, 0.5266]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0134
95% CI (Adjusted): [0.4750, 0.5276]
Gender: Male

Count: 748
Proportion (p): 0.4987
Standard Error (SE): 0.0129
95% CI (Unadjusted): [0.4734, 0.5240]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0134
95% CI (Adjusted): [0.4724, 0.5250]
Income Level: Middle

Count: 761
Proportion (p): 0.5073
Standard Error (SE): 0.0129
95% CI (Unadjusted): [0.4820, 0.5326]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0134
95% CI (Adjusted): [0.4810, 0.5336]
Income Level: Low

Count: 447
Proportion (p): 0.2980
Standard Error (SE): 0.0118
95% CI (Unadjusted): [0.2749, 0.3211]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0123
95% CI (Adjusted): [0.2739, 0.3221]
Income Level: High

Count: 292
Proportion (p): 0.1947
Standard Error (SE): 0.0102
95% CI (Unadjusted): [0.1746, 0.2147]
Intraclass Correlation (rho): 0.02
Average Cluster Size (m): 5
Design Effect (DEFF): 1.0800
Adjusted Standard Error (SE_adj): 0.0106
95% CI (Adjusted): [0.1738, 0.2155]
'''