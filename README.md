# my-girlfriend-is-a-SPY
A Logistic Regression based "SATIRE" ML Model to identify if your girlfriend is a SPY.

# Spy Classification Model 🕵️‍♂️🤖

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Environment:** Jupyter Notebook / JupyterLab (`train_model.ipynb`)
* **Libraries:** 
  * `pandas` for data handling and inspection
  * `scikit-learn` for data splitting, logistic regression modeling, and performance evaluation metrics

---

## 📈 Methodology & Pipeline

1. **Data Inspection:**
   * Loaded `spy_dataset.csv` containing 9 total features: `two_phones`, `multiple_aliases`, `company_valuation_above_500B`, `works_for_gov`, `jokes_about_kim_jong_un`, `unexpected_travel_frequency`, `knows_cryptography`, `avoids_taiwan_topic`, and the target `is_spy`.
   * Verified non-null completeness across all 1,000 rows.

2. **Data Splitting:**
   * Separated features ($X$) and target ($y$).
   * Performed a train-test split with a 67/33 ratio (`test_size=0.33`, `random_state=42`), resulting in a heavily skewed training distribution (636 legitimate vs. 34 spy instances).

3. **Model Training:**
   * Initialized and trained a `LogisticRegression` classifier with `max_iter=10000` to ensure complete algorithmic convergence.

4. **Evaluation Results:**
   * **Overall Accuracy:** ~95.15% (artificially inflated due to majority class dominance)
   * **Classification Report:**
     * **Class 0 (Non-Spy):** Precision: 0.95 | Recall: 1.00 | F1-Score: 0.98
     * **Class 1 (Spy):** Precision: 0.50 | Recall: 0.06 | F1-Score: 0.11
