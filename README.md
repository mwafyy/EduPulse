# EduPulse – AI-Powered Student Early Warning System

EduPulse is a web-based educational monitoring system that uses Machine Learning to identify students who may be at risk of academic failure.

The idea behind the project is to help teachers identify potential academic problems early, understand student risk levels, and organize follow-up actions before the end of the academic year.

The system combines a Machine Learning pipeline, a FastAPI backend, a web dashboard, and role-based access control.

## 1. Project Overview

EduPulse focuses on early identification rather than waiting for final academic results.

The prediction pipeline uses student information that can be available during the academic year, including:

- Absences
- Previous academic failures
- Study time
- Family and school support
- Extracurricular activities
- Other relevant student characteristics

Grades from the first, second, and final periods (G1, G2, G3) are excluded from the prediction features to reduce data leakage and make the prediction scenario closer to an early-warning use case.

## 2. Machine Learning

The project includes a complete Machine Learning workflow covering data exploration, preprocessing, model training, comparison, evaluation, and prediction.

The evaluated models include:

- Dummy Classifier as a baseline.
- Balanced Logistic Regression.
- Random Forest Classifier.

Random Forest was selected as the main model based on the evaluation results.

Reported evaluation metrics include:

- Recall: 69.6%
- ROC-AUC: 0.827
- Stratified 5-Fold Cross-Validation

Recall is particularly important in this project because missing a student who is actually at risk can delay the necessary support.

The trained pipelines are saved using Joblib and loaded by the prediction module, so the application can make predictions without retraining the models every time it starts.

## 3. Student Risk Assessment

EduPulse classifies students into three risk categories:

- Low Risk
- Medium Risk
- High Risk

Each assessment includes a risk probability that helps teachers prioritize students who may need attention.

The application also provides an individual prediction workflow using student information.

## 4. Dashboard and Student Management

The dashboard brings student monitoring and follow-up activities together in one interface.

Current functionality includes:

- Dashboard summary and key performance indicators.
- Student distribution across risk categories.
- Student search and filtering.
- Individual student profiles and risk assessments.
- Risk-related factors and model information.
- Suggested intervention actions.
- Intervention records and follow-up status.
- Individual student prediction.

The purpose is to make model outputs useful in an educational workflow, rather than presenting predictions as isolated numbers.

## 5. Authentication and Access Control

The backend implements authentication and role-based permissions.

**Administrator**
- Can access students across all subjects.
- Can review the overall dashboard.
- Has access to the full student data scope.

**Teacher**
- Can access students assigned to their account and subject.
- Cannot access another teacher's student records.
- Cannot modify, delete, or create records outside their assigned scope.
- Can manage interventions only for students within that scope.

The current local setup includes separate teacher accounts for Math and Portuguese, alongside an administrator account.

Authentication and password security are implemented using:

- JWT access tokens.
- Argon2 password hashing.
- Backend-side authorization checks.

Data isolation is enforced by the backend. It does not rely only on hiding information in the frontend.

## 6. Technology Stack

| Component | Technologies |
|---|---|
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Authentication | JWT, Argon2 |
| Database | SQLite |
| API Documentation | Swagger UI |

## 7. Project Structure

```text
EduPulse/
│
├── backend/
│   ├── main.py
│   └── database.py
│
├── frontend/
│   └── index.html
│
├── ml/
│   ├── predict.py
│   ├── train.py
│   └── saved_models/
│       ├── lr_binary_pipeline.joblib
│       ├── rf_binary_pipeline.joblib
│       ├── rf_multiclass_pipeline.joblib
│       └── metrics_and_metadata.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

The repository contains the application source code and trained model artifacts. Local database files, backups, and raw student datasets are kept outside the repository.

## 8. Local Setup

### Requirements

- Python 3.14.6
- pip

### Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd EduPulse
```

### Install dependencies

```bash
py -m pip install -r requirements.txt
```

### Configure the environment

The application requires a secure environment variable named `JWT_SECRET_KEY`.

Generate a strong random secret and configure it in your local environment. Never commit the secret or any `.env` file to the repository.

### Database configuration

The current development setup uses SQLite.

The application expects an existing, properly initialized database at:

```text
data/edupulse.db
```

The local database is intentionally excluded from the repository because it contains application records and should not be published.

A fresh clone therefore requires a separately prepared database before the application can be used.

### Run the application

```bash
py -m uvicorn backend.main:app --host 127.0.0.1 --port 8002
```

Dashboard:

```text
http://127.0.0.1:8002/
```

API documentation:

```text
http://127.0.0.1:8002/docs
```

## 9. Security and Data Handling

Student records and database backups are not included in the Git repository.

The project follows these practices:

- Passwords are hashed instead of being stored as plain text.
- Protected API operations require authentication.
- Teacher permissions are checked on the backend.
- Student access is restricted according to teacher assignments.
- Secret keys are configured outside the source code.
- Local database files and raw datasets are excluded from version control.

## 10. Development Status

The current version includes a working local application with:

- FastAPI backend and web dashboard.
- Saved Machine Learning prediction pipelines.
- Student risk classification.
- Administrator and teacher roles.
- Subject-based data isolation.
- Student management and intervention tracking.
- SQLite database integration.

Future development work includes preparing a production deployment, migrating from SQLite to PostgreSQL, and configuring production hosting and environment settings.

## 11. Project Objective

EduPulse explores how Machine Learning can support educational decision-making.

The goal is not simply to predict academic failure, but to connect data analysis and prediction with practical actions that help teachers monitor students and provide support when it is needed.
