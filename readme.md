# 📊 Telco Customer Churn: Power BI Dashboard & Python Data Science Project

Hi! 👋 Welcome to my end-to-end **Telco Customer Churn Analysis & Machine Learning Project**. 

As a fresher building my data analytics and data science portfolio, I wanted to take a real-world telecom dataset and solve a business problem from two angles:
1. **Business Intelligence (Power BI):** I built my first interactive **Power BI Dashboard** to visualize customer churn KPIs and uncover high-risk customer segments at a glance.
2. **Data Science & Machine Learning (Python Jupyter Notebook):** I followed the complete data science lifecycle in Python—cleaning the raw data, validating my dashboard insights with statistical Exploratory Data Analysis (EDA), engineering features, and training Machine Learning models to predict which customers are most likely to churn.

---

## 🖥️ My Power BI Dashboard

<img width="1846" height="1061" alt="screenshot-1793191540603" src="https://github.com/user-attachments/assets/e6984c2a-ec7d-4992-9457-7629ea7ad785" />



### What I Built in This Dashboard:
- **Interactive Slicers:** Added filters for `InternetService` (*DSL*, *Fiber optic*, *No*) and `Contract` (*Month-to-month*, *One year*, *Two year*) so stakeholders can dynamically slice every chart on the page.
- **Top-Level KPI Cards:** Highlighted **7,043 Total Customers** and **1,869 Churned Customers** right at the top.
- **Visual Breakdowns:** Designed 8 clean visuals using a consistent two-tone palette (**Steel Blue** for Retained customers and **Orange-Red** for Churned customers) to compare churn across **Contract Type**, **Average Monthly Charges**, **Partner Status**, **Dependents**, **Gender**, **Tech Support**, and **Payment Method**.

---

## 🎯 Problem Statement & Objective

Customer churn (when a customer cancels their subscription and leaves for a competitor) is one of the biggest revenue leaks for telecom companies. Acquiring a new customer costs **5x to 7x more** than retaining an existing one.

**My Objectives in This Project:**
1. Understand **why** customers are leaving by analyzing demographics, billing methods, contract terms, and subscribed services.
2. Quantify the exact churn rate across different customer segments in both **Power BI** and **Python**.
3. Build and compare **Machine Learning classification models** (`Logistic Regression`, `Decision Tree`, and `Random Forest`) to predict customer churn before it happens.
4. Provide **actionable business recommendations** that a retention team could realistically use to reduce churn.

---

## 🛠️ Tools & Technologies Used

- **BI & Visualization:** Power BI Desktop
- **Programming Language:** Python 3
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization (Python):** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `StandardScaler`, `train_test_split`, `roc_auc_score`, `confusion_matrix`)
- **Environment:** Jupyter Notebook (`.ipynb`)

---

## 🔍 Step-by-Step Data Science Process (In My Jupyter Notebook)

### Step 1: Data Loading & Initial Inspection
- Loaded `WA_Fn-UseC_-Telco-Customer-Churn.csv` containing **7,043 rows (customers)** and **21 columns (features)**.
- Checked data types using `df.info()` and noticed that `TotalCharges` was stored as an `object` (text string) instead of a `float64` number.

### Step 2: Data Cleaning & Preprocessing Checks
- **Fixing `TotalCharges`:** When I investigated why `TotalCharges` was text, I found **11 rows** containing blank spaces (`" "`). By inspecting those 11 customers, I discovered they all had `tenure == 0` and `Churn == "No"`—meaning they had just signed up and hadn't completed their first billing cycle yet! I converted the column using `pd.to_numeric(..., errors='coerce')` and filled those 11 missing values with `0.0`.
- **Readable Labels for EDA:** Mapped `SeniorCitizen` from `0`/`1` to `"No"`/`"Yes"` so my EDA charts are easy to read.
- **Duplicate & ID Check:** Verified there are **0 duplicate rows** and dropped `customerID` before modeling since unique IDs have no predictive power.

### Step 3: Exploratory Data Analysis (EDA) — Key Insights
I recreated and expanded my Power BI visuals in Python using `matplotlib` and `seaborn` to verify every metric:

| Customer Segment / Metric | Finding from My Analysis | Business Interpretation |
| :--- | :--- | :--- |
| **Overall Churn Rate** | **26.54%** (`1,869` Churned vs. `5,174` Retained) | Roughly 1 in 4 customers leaves; dataset is moderately imbalanced (~73.5% vs ~26.5%). |
| **Contract Type** | **Month-to-month:** `42.71%` churn<br>**One year:** `11.27%` churn<br>**Two year:** `2.83%` churn | Month-to-month customers have no long-term lock-in and churn **15x more** than 2-year contract users. |
| **Monthly Charges** | **Churned Avg:** `$74.44` (Median `$79.65`)<br>**Retained Avg:** `$61.27` (Median `$64.43`) | Customers who leave are paying roughly **$13/month more** on average (driven heavily by Fiber Optic plans). |
| **Tech Support** | **Without Tech Support:** `41.64%` churn<br>**With Tech Support:** `15.17%` churn<br>**No Internet:** `7.40%` churn | Internet users without Tech Support churn at nearly **3x the rate** of supported users. |
| **Payment Method** | **Electronic check:** `45.29%` churn<br>**Mailed check:** `19.11%` churn<br>**Bank transfer (auto):** `16.71%` churn<br>**Credit card (auto):** `15.24%` churn | Customers paying manually via Electronic Check have the highest churn of any payment group. |
| **Partner & Dependents** | **No Partner:** `32.96%` (vs. `19.66%` with Partner)<br>**No Dependents:** `31.28%` (vs. `15.45%` with Dependents) | Single customers switch providers much more easily than family/household accounts. |
| **Customer Tenure** | **Churned Median Tenure:** `10 months`<br>**Retained Median Tenure:** `38 months` | The highest churn risk occurs in the **first 6 to 12 months** of a customer's lifecycle. |
| **Gender** | **Female:** `26.92%` churn \| **Male:** `26.16%` churn | Gender has practically zero effect on whether a customer churns. |

### Step 4: Feature Engineering & Data Preparation
- Converted binary `"Yes"`/`"No"` columns into `1`/`0`.
- Applied **One-Hot Encoding** (`pd.get_dummies(..., drop_first=True)`) to multi-category columns (`Contract`, `InternetService`, `PaymentMethod`, etc.), resulting in **30 clean numerical features**.
- Split the dataset into **80% Training (`5,634` customers)** and **20% Testing (`1,409` customers)** using `stratify=y` so both sets have the exact same `26.54%` churn ratio.
- Scaled continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` (fitted **only** on the training data to prevent data leakage).

### Step 5: Machine Learning Model Comparison
I trained three classification models on the training set and evaluated them on the unseen test set (`1,409` customers):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **81.26%** | **69.12%** | **53.48%** | **60.24%** | 85.74% |
| **Random Forest** | 80.27% | 69.51% | 43.85% | 53.82% | **85.78%** |
| **Decision Tree** | 79.21% | 64.10% | 47.59% | 54.71% | 84.02% |

**Why I Picked Logistic Regression as My Best Baseline Model:**
While Random Forest achieved a slightly higher ROC-AUC (`85.78%`), **Logistic Regression** achieved the highest **Accuracy (`81.26%`)**, highest **Recall (`53.48%`)**, and highest **F1-Score (`60.24%`)**. In churn prediction, **Recall** is critical because missing an actual churner (False Negative) means losing a customer permanently.

---

## 💡 My Business Recommendations to Reduce Churn

Based on both my Power BI dashboard and my Python model's top feature importances (`tenure`, `InternetService_Fiber optic`, `Contract_Two year`, `PaymentMethod_Electronic check`, and `TechSupport`), here are four actionable steps the company should take:

1. **Convert Month-to-Month Users to 1-Year Contracts (`42.7%` → `11.3%` churn):** Offer a targeted 10%–15% loyalty discount or one free month at the 3-month mark to nudge month-to-month customers onto a 1-year contract.
2. **Bundle Free Tech Support Trials with Fiber Optic Plans (`41.6%` → `15.2%` churn):** Since Fiber Optic customers pay higher monthly bills (`~$74+`) and churn heavily when they lack support, include a **3-month free Tech Support + Online Security trial** for new Fiber Optic sign-ups.
3. **Nudge Electronic Check Users Toward Auto-Pay (`45.3%` → `~16%` churn):** Provide a small `$5/month` bill credit for customers who switch from manual Electronic Checks to automatic Bank Transfer or Credit Card billing.
4. **Focus Onboarding Retention on Months 1–6:** Use the Logistic Regression model's predicted churn probabilities every month to flag high-risk customers during their first 6 months and trigger proactive customer care check-ins.

---

## 📁 Repository Structure

```text
├── README.md                              # Project documentation & findings summary
├── Telco_Customer_Churn_Project.ipynb     # Full Python Data Science & ML Jupyter Notebook (pre-executed)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw Telco Customer Churn dataset (7,043 rows, 21 columns)
└── dashboard_preview.jpg                  # Screenshot of my interactive Power BI dashboard
```

---

## 🚀 How to Run This Project Locally

1. **Clone this repository:**
   ```bash
   git clone https://github.com/<your-username>/telco-customer-churn-analysis.git
   cd telco-customer-churn-analysis
   ```
2. **Install the required Python libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Telco_Customer_Churn_Project.ipynb
   ```

---

*Thank you for checking out my project! Feedback and suggestions are always welcome.* 😊
