# Customer Churn Prediction - 21AIC401T
This project aims to predict customer churn for a telecom company using machine learning techniques. The dataset used for this project contains information about customers, their demographics, and their usage patterns.

## Project Structure
- `data/`: Contains the raw and processed data.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `src/`: Source code for data processing, feature engineering, and model training.
- `app/`: Flask web application for serving the model.
- `outputs/`: Generated plots and model artifacts.

## Requirements
- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

## Usage
1. Install the required packages:
```
pip install -r requirements.txt
```
2. Run the Flask app:
```
python -m flask run
```
3. Send a POST request to the `/predict` endpoint with the customer data:
```
curl -X POST -H "Content-Type: application/json" -d '{"customer_id": 12345, "age": 30, "gender": "M", "tenure": 12, "monthly_charges": 70.0}' http://localhost:8080/predict
```

## Model Training
The model is trained using the script `churn_project.ipynb` which:
1. Loads the data
2. Preprocesses the data (handle missing values, encode categorical variables, etc.)
3. Splits the data into training and testing sets
4. Trains a logistic regression and a decision tree
5. Evaluates the model and saves artifacts
