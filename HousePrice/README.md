# House Price Prediction using Linear Regression

A machine learning project that builds and optimizes a Linear Regression model to predict residential house prices. The project cleans raw housing features, applies categorical encoding, handles extreme data outliers, and scales variables to optimize model convergence.

## 📊 Model Performance Summary

After cleaning and filtering the dataset, the model's accuracy increased significantly:

* **Initial R² Score:** 0.10 (10% variance explained)
* **Optimized R² Score:** 0.56 (56% variance explained)
* **Model Intercept (Baseline Price):** \$456,422.38
* **Final Mean Squared Error (MSE):** 1.84e+10

### Key Insights from Feature Analysis
1. **Location Dominance:** Specific geographic locations (one-hot encoded cities) add the highest mathematical weight to the price, with top locations adding over +\$122k to a home's value.
2. **Living Space:** Square footage of the living area (`sqft_living`) is the strongest physical driver of overall valuation.
3. **Diminishing Returns on Bedrooms:** Holding square footage constant, adding extra bedrooms results in a minor negative coefficient weight, indicating that splitting layout space into smaller rooms lowers baseline property value.

---

## 📈 Evaluation Visualization

The plot below visualizes the relationship between the actual market prices and the model's predicted prices on the test data. The red dashed diagonal represents a hypothetical perfect match.

![Actual Prices vs Model Predictions](https://githubusercontent.com) 
*(Note: Replace this link with your actual image path once uploaded to GitHub)*

The scatter plot displays a strong, linear alignment around the ideal prediction line for mid-range homes, confirming that outlier removal successfully stabilized the model's regression line.

---

## 🛠️ Data Pipeline & Engineering Steps

The project implements a structured pipeline to prepare the dataset for Linear Regression:

1. **Irrelevant Feature Removal:** 
   Dropped text strings and geographic identifiers like `date`, `street`, `statezip`, and `country`.
2. **Feature Transformation (Ages instead of Years):**
   * Transformed `yr_built` into `house_age` using the dataset reference frame.
   * Transformed `yr_renovated` into a structural `renovation_age` feature using efficient vectorization (`np.where`).
3. **Categorical Handling (One-Hot Encoding):**
   Converted the `city` column using `pd.get_dummies(drop_first=True)` to prevent the model from treating independent text categories as ordered numbers.
4. **Outlier Mitigation:**
   Removed high-leverage luxury anomalies and massive rural plots (`price < $1,000,000` and `sqft_lot < 20,000`) that heavily skewed the regression line.
5. **Feature Scaling:**
   Applied `StandardScaler` to bring small scale features (like `bedrooms`) and large scale features (like `sqft_living`) onto a uniform matrix distribution.

---