
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the Picard Iterative Method (PMIM)
def picard_iterative_method(func, x0, iterations, step_size):
    x_values = [x0]
    for i in range(1, iterations):
        x_new = x_values[-1] + step_size * func(x_values[-1])
        x_values.append(x_new)
    return np.array(x_values)

# Function to visualize the result
def plot_results(iterations, x_values):
    plt.plot(range(iterations), x_values, label="Picard Iteration")
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.title('Picard Iterative Method')
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
st.title("PicardPredict: Revolutionizing Predictive Modelling with Iterative Precision")

st.sidebar.header("Input Parameters")

# User inputs for Picard Iterative Method
initial_condition = st.sidebar.number_input("Initial Condition (x0)", value=0.0)
iterations = st.sidebar.slider("Number of Iterations", min_value=1, max_value=100, value=20)
step_size = st.sidebar.number_input("Step Size", value=0.1)

# Choose a function to apply Picard Iterative Method on
st.sidebar.header("Choose a Function")
func_choice = st.sidebar.selectbox("Select Function", ["Exponential Growth", "Sinusoidal", "Linear"])

if func_choice == "Exponential Growth":
    def func(x): return np.exp(x)
elif func_choice == "Sinusoidal":
    def func(x): return np.sin(x)
else:
    def func(x): return 0.5 * x

# Compute the result using Picard Iterative Method
x_values = picard_iterative_method(func, initial_condition, iterations, step_size)

# Display the result
st.write(f"Results for {func_choice} function with initial condition {initial_condition}, {iterations} iterations, and step size {step_size}:")
plot_results(iterations, x_values)

# Display the final value of the iteration
st.write(f"Final Value after {iterations} iterations: {x_values[-1]}")

# GitHub link for the repository
st.sidebar.write("[Visit GitHub Repository](https://github.com/yourusername/PicardPredict)")
