# 💳 CreditWise — Loan Approval Prediction System

Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval.
Implemented Binary classification along with EDA, feature engineering & model evaluation (Precision, Recall, F1).

---

## 📌 Project Overview

Banks and financial institutions receive thousands of loan applications daily. This project automates the loan approval decision by training classification models on historical applicant data. The best-performing model is selected based on precision score.

---

## 🏦 Problem Statement

**SecureTrust Bank** — a mid-sized financial company — offers personal and home loans to customers across urban and rural regions. Every day, hundreds of customers apply for loans through online and branch applications.

Until now, the bank has been using a **manual verification process** where loan officers evaluate applications by checking income proofs, employment details, credit history, and other documents. This process is time-consuming, biased & inconsistent.

**Two major challenges faced by the bank:**
1. Good customers sometimes get **rejected** → loss of business
2. High-risk customers sometimes get **approved** → financial losses

**Solution:** An intelligent ML-powered system that automatically analyses applicant details and predicts whether a loan should be **Approved or Rejected** before final human verification.

---

## 🗂️ Dataset

**File:** `loan_approval_data.csv`

Each row represents a **loan applicant** with personal, financial, and credit information.

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant ID |
| `Applicant_Income` | Monthly income of the applicant |
| `Coapplicant_Income` | Monthly income of the co-applicant |
| `Employment_Status` | Salaried / Self-Employed / Business |
| `Age` | Age of the applicant |
| `Marital_Status` | Married / Single |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of already running loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Loan amount requested |
| `Loan_Term` | Loan duration (months) |
| `Loan_Purpose` | Home / Education / Personal / Business |
| `Property_Area` | Urban / Semi-Urban / Rural |
| `Education_Level` | Graduate / Postgraduate / Undergraduate |
| `Gender` | Male / Female |
| `Employer_Category` | Govt / Private / Self |
| `Loan_Approved` | **Target** — 1 = Approved, 0 = Rejected |

---

## 🔧 Project Pipeline

### 1. Data Loading & Exploration
- Loaded dataset using `pandas`
- Checked `.head()`, `.describe()`, `.info()`, and `.isnull().sum()` for initial understanding

### 2. Handling Missing Values
- Numerical columns → imputed with **mean** using `SimpleImputer`
- Categorical columns → imputed with **most frequent** value

### 3. Exploratory Data Analysis (EDA)
- Pie chart and bar plot for class balance of `Loan_Approved`
- Count plots for all categorical features
- Histograms for `Applicant_Income` and `Coapplicant_Income`
- Box plots for outlier detection across numerical features vs target
- `Credit_Score` distribution by loan approval status

### 4. Feature Encoding
- `LabelEncoder` for `Education_Level` and `Loan_Approved`
- `OneHotEncoder` (drop first) for: `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`

### 5. Correlation Analysis
- Heatmap of full correlation matrix
- Top correlated features with `Loan_Approved` ranked

### 6. Train-Test Split & Feature Scaling
- 80/20 train-test split with `random_state=42`
- `StandardScaler` applied (fit on train, transform on test)

### 7. Model Training & Evaluation
Three classifiers were trained and compared:

| Model | Metric Used |
|---|---|
| Logistic Regression | Precision, Recall, F1, Confusion Matrix |
| KNN (k=5) | Precision, Recall, F1, Confusion Matrix |
| Naive Bayes (GaussianNB) | Precision, Recall, F1, Confusion Matrix |

> ✅ **Best Model: Naive Bayes** — highest precision score among all three

### 8. Feature Engineering
- Added `DTI_Ratio_sq = DTI_Ratio²` and `Credit_Score_Sq = Credit_Score²`
- Retrained Naive Bayes on engineered features to further enhance performance

---

## 📊 Results

| Model | Winner |
|---|---|
| Logistic Regression | — |
| KNN Classifier | — |
| **Naive Bayes** | ✅ Best Precision |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- **Environment:** Jupyter Notebook

---

## 📁 Project Structure

```
CreditWise-Loan-Approval/
│
├── credit_wise_loan_project.ipynb   # Main notebook
├── loan_approval_data.csv           # Dataset
└── README.md                        # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/tal123ph/creditwise-loan-approval.git
   cd creditwise-loan-approval
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook credit_wise_loan_project.ipynb
   ```

4. Make sure `loan_approval_data.csv` is in the same directory as the notebook.

---

## 👤 Author

**Talha**
- GitHub: [@tal123ph](https://github.com/tal123ph)
- LinkedIn: [Muhammad Talha](https://www.linkedin.com/in/muhammad-talha12b/)
- Student — Data Analytics, University of Agriculture Faisalabad (UAF)
- AWS AI & ML Scholars Programme

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
