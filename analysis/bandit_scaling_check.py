import numpy as np

def analyze_scaling(q_val, other_val, label):
    features = np.array([q_val, other_val])
    features_sum = features.sum()
    if features_sum == 0:
        normalized = np.array([np.nan, np.nan])
    else:
        normalized = features / features_sum
    
    print(f"--- {label} ---")
    print(f"Raw: {features}")
    print(f"Sum: {features_sum}")
    print(f"Normalized (Sum-to-1): {normalized}")
    print(f"Magnitude preserved? No.")
    print(f"Ratio preserved? {normalized[0]/normalized[1] if normalized[1]!=0 else 'inf'}")
    print("")

print("=== Analyzing BALD Features Scaling ===")
# Case 1: Low Information (Model confident)
analyze_scaling(0.001, 0.002, "Low Info (0.001, 0.002)")
# Case 2: High Information (Model uncertain)
analyze_scaling(1.0, 2.0, "High Info (1.0, 2.0)")

# Case 3: Zero (Potential Bug)
analyze_scaling(0.0, 0.0, "Zero Info (0.0, 0.0)")


print("=== Analyzing Vendi Features Scaling ===")
# Vendi Score (Diversity). 
# Case A: Small diversity (similar samples)
analyze_scaling(2.0, 5.0, "Low Vendi (2.0, 5.0)")
# Case B: Large diversity (diverse samples)
analyze_scaling(20.0, 50.0, "High Vendi (20.0, 50.0)")

print("\n=== Linearity Check ===")
norm = np.array([0.333, 0.667])
bias = 1.0
t = 0.5
print(f"Features: [{norm[0]}, {norm[1]}, {t}, {bias}]")
print(f"Constraint: x[0] + x[1] = {norm[0]+norm[1]}")
print("This shows perfect collinearity between feature 0, feature 1, and the bias term if bias=1.")

