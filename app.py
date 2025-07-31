import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Picard Iterative Method
def picard_iterative_method(func, x0, iterations, step_size):
    x_values = [x0]
    for i in range(1, iterations):
        x_new = x_values[-1] + step_size * func(x_values[-1])
        x_values.append(x_new)
    return np.array(x_values)

# Runge-Kutta 4th order method
def runge_kutta(func, x0, t0, tf, step_size):
    n_steps = int((tf - t0) / step_size)
    x_values = [x0]
    t_values = np.arange(t0, tf, step_size)

    for i in range(n_steps):
        k1 = step_size * func(x_values[-1], t_values[i])
        k2 = step_size * func(x_values[-1] + 0.5 * k1, t_values[i] + 0.5 * step_size)
        k3 = step_size * func(x_values[-1] + 0.5 * k2, t_values[i] + 0.5 * step_size)
        k4 = step_size * func(x_values[-1] + k3, t_values[i] + step_size)

        x_new = x_values[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_values.append(x_new)

    return np.array(x_values), t_values

# Plot results function
def plot_results(iterations, x_values, method="Picard Iteration"):
    plt.plot(range(iterations), x_values, label=f"{method} Results")
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.title(f'{method} - Prediction')
    plt.legend()
    st.pyplot(plt)

# Validate custom equation
def validate_custom_equation(equation_input):
    try:
        # Convert the user input string into a lambda function
        func = sp.sympify(equation_input)
        return func
    except:
        return None

# Streamlit UI
st.title("PicardPredict: Revolutionizing Predictive Modelling with Iterative Precision")

# Step 1: Enter Data for Prediction
st.sidebar.header("Step 1: Enter Data for Prediction")

# User input for the differential equation and initial conditions
equation_choice = st.sidebar.selectbox("Choose the Differential Equation", ["Custom Equation", "Exponential Growth", "Sinusoidal", "Linear"])
initial_condition = st.sidebar.number_input("Initial Condition (x0)", value=0.05)
iterations = st.sidebar.slider("Number of Iterations", min_value=1, max_value=100, value=7)
step_size = st.sidebar.number_input("Step Size", value=0.01)

# Handle custom equation input
if equation_choice == "Custom Equation":
    equation_input = st.sidebar.text_area("Enter your custom equation as a function of x (e.g., x**2, np.sin(x), etc.):")
    func = validate_custom_equation(equation_input)
    if func is None:
        st.sidebar.error("Invalid equation input! Please use a valid Python expression.")
else:
    if equation_choice == "Exponential Growth":
        func = lambda x: np.exp(x)
    elif equation_choice == "Sinusoidal":
        func = lambda x: np.sin(x)
    else:
        func = lambda x: 0.5 * x

# Step 2: Solve using Picard Iterative Method
if st.sidebar.button("Solve using Picard Method"):
    if func:
        picard_results = picard_iterative_method(func, initial_condition, iterations, step_size)
        plot_results(iterations, picard_results, "Picard Iteration")

        # Step 3: Compare with Runge-Kutta (RK4) Method
        tf = 10  # Final time for RK4
        t0 = 0  # Initial time for RK4
        rk4_results, t_values = runge_kutta(func, initial_condition, t0, tf, step_size)

        # Plot RK4 Results
        plt.plot(t_values, rk4_results, label="RK4 Method", color='red')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Runge-Kutta (RK4) Method - Comparison')
        plt.legend()
        st.pyplot(plt)

        # Show comparison details
        st.write(f"Picard Method Final Value: {picard_results[-1]}")
        st.write(f"RK4 Method Final Value: {rk4_results[-1]}")

        # Step 4: Display the Graph and Solution
        st.write("Results have been displayed successfully!")
    else:
        st.sidebar.error("Invalid equation! Please fix your custom equation input.")

# Link to GitHub Repository
st.sidebar.write("[Visit GitHub Repository](https://github.com/yourusername/PicardPredict)")
