# Customer Churn Prediction

## Dataset

This project uses a customer behavior dataset containing demographic, engagement, purchasing, and service interaction features.

The dataset includes information such as customer profile details, platform usage patterns, purchase behavior, communication engagement, and customer service interactions.

Dataset source:
> https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset

---

# Project Goal

The goal of this project is to analyze customer behavior patterns and build a machine learning model capable of predicting whether a customer is likely to churn.

The project focuses on:

- Understanding customer behavior through exploratory data analysis.
- Identifying important factors associated with customer churn.
- Building and evaluating classification models.
- Creating an explainable ML system using SHAP.
- Deploying the final model through an API.

---

# Project Structure

```

Customer-Churn-Prediction/

│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_analytics.ipynb
│   └── 03_model.ipynb
│
├── app/
│   └── main.py
│
├── churn_model.pkl
├── requirements.txt
├── Dockerfile
└── README.md

````

---

# Notebooks

## 01_eda.ipynb

Data understanding and preprocessing:

- Dataset overview
- Missing value analysis
- Data quality checks
- Handling invalid values
- Feature inspection


## 02_analytics.ipynb

Customer behavior analysis:

- Feature relationships
- Customer segmentation analysis
- Churn pattern exploration
- Business-focused insights


## 03_model.ipynb

Machine learning workflow:

- Feature selection
- Data preprocessing pipeline
- Model training
- Cross-validation and tuning
- Model evaluation
- Threshold analysis
- SHAP-based model explainability


---

# Installation

## Option 1: Run Locally

Clone the repository:

```bash
git clone <github-repository-link>

cd Customer-Churn-Prediction
````

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Option 2: Run Using Docker

Build the Docker image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 churn-api
```

API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* FastAPI
* Docker

```

This is intentionally kept small and portfolio-friendly. It explains **what the project is, how to run it, and how it is structured** without dumping model results or turning the README into a report.
```

