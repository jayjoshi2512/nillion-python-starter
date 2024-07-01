# Import necessary libraries
from nada_dsl import *
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score

# Define the main function
def nada_main():
    # Define two parties
    party1 = Party(name="Party1")
    party2 = Party(name="Party2")
    
    # Define secret inputs for each party
    data1 = SecretArray(Input(name="data1", party=party1), shape=(100, 2))
    data2 = SecretArray(Input(name="data2", party=party2), shape=(100, 2))
    
    # Combine the data securely
    combined_data = SecretArray.vstack([data1, data2])
    
    # Split the data into features (X) and target (y)
    X = combined_data[:, :-1]
    y = combined_data[:, -1]
    
    # Normalize the features securely
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.reveal())
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_normalized)
    
    # Perform secure linear regression with cross-validation
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y.reveal(), cv=5)
    
    # Fit the model on the entire data
    model.fit(X_poly, y.reveal())
    
    # Get the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Calculate the mean cross-validation score
    mean_cv_score = np.mean(scores)
    
    # Secure scalar multiplication (dummy example)
    scalar = 3
    secure_scalar_product = scalar * SecretScalar(y.sum())
    
    # Securely output the model parameters and evaluation metrics
    return [
        Output(coefficients, "coefficients_output", [party1, party2]),
        Output(intercept, "intercept_output", [party1, party2]),
        Output(mean_cv_score, "mean_cv_score_output", [party1, party2]),
        Output(secure_scalar_product, "secure_scalar_product_output", [party1, party2])
    ]

# Define a function to plot results using Plotly
def plot_results(X, y, model, title="Linear Regression Results"):
    # Predict the values
    X_poly = PolynomialFeatures(degree=2).fit_transform(X)
    y_pred = model.predict(X_poly)
    
    # Create a scatter plot of the data
    scatter = go.Scatter(x=X[:, 0], y=y, mode='markers', name='Data')
    
    # Create a line plot of the predictions
    line = go.Scatter(x=X[:, 0], y=y_pred, mode='lines', name='Prediction')
    
    # Create the figure
    fig = go.Figure(data=[scatter, line])
    
    # Update layout
    fig.update_layout(title=title, xaxis_title='Feature', yaxis_title='Target')
    
    # Show the figure
    fig.show()

# Now create a main function to execute the program
if __name__ == "__main__":
    # Simulate input data
    data1 = np.random.rand(100, 2)
    data2 = np.random.rand(100, 2)
    
    # Define inputs for the simulation
    inputs = [
        Input(name="data1", party=party1, value=data1),
        Input(name="data2", party=party2, value=data2)
    ]
    
    # Execute the MPC program
    results = execute_nada_program(nada_main, inputs)
    
    # Print the results
    print("Coefficients:", results["coefficients_output"])
    print("Intercept:", results["intercept_output"])
    print("Mean CV Score:", results["mean_cv_score_output"])
    print("Secure Scalar Product:", results["secure_scalar_product_output"])
    
    # Plot the results
    combined_data = np.vstack([data1, data2])
    X = combined_data[:, :-1]
    y = combined_data[:, -1]
    
    # Normalize and create polynomial features for plotting
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    model = LinearRegression()
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_normalized)
    model.fit(X_poly, y)
    
    plot_results(X_normalized, y, model)
