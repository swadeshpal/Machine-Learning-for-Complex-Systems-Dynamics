# -*- coding: utf-8 -*-
"""
Example 1: Estimation of equilibrium points and saddle-node bifurcation
using a neural-network-based residual minimization approach.

The code approximates the bifurcation parameter r as a function of the
state variable u. The neural network is trained so that the governing
equilibrium equation is satisfied over the selected state-variable range.

After training:
    1. The predicted r(u) relationship is evaluated.
    2. Local extrema of r(u) are identified as candidate turning points.
    3. The Jacobian eigenvalue is used to classify the computed
       equilibrium points as stable or unstable.
    4. The results are plotted and exported to text files.

Note:
The original mathematical model and computational workflow are retained.
"""

# ============================================================================
# SECTION 1: IMPORTING REQUIRED LIBRARIES
# ============================================================================

import torch                         # Tensor operations and automatic differentiation
from torch import nn                 # Neural-network layers and loss functions
import matplotlib.pyplot as plt     # Plotting graphs and bifurcation diagrams
import numpy as np                   # Numerical arrays and data processing
import torch.autograd as autograd    # Automatic differentiation utilities
import sympy as sp                   # Symbolic differentiation and eigenvalue calculation


# ============================================================================
# SECTION 2: GENERATING THE INPUT/TRAINING DATA
# ============================================================================

# Define the interval of the state variable u over which the network is trained.
start = 0.0                           # Lower bound of the state-variable interval
end = 1.5                             # Upper bound of the state-variable interval

# Number of equally spaced training points in the selected interval.
steps = 1201

# Generate uniformly distributed points between start and end.
ut = np.linspace(start, end, steps)

# Convert the NumPy array into a PyTorch tensor.
# The reshape operation gives the input tensor the shape (number of points, 1).
u_train = torch.Tensor(ut).reshape(-1, 1)


# ============================================================================
# SECTION 3: SPECIFYING TRAINING PARAMETERS AND NETWORK ARCHITECTURE
# ============================================================================

num_iter = 50000                      # Total number of neural-network training iterations
interval_iter = 5000                  # Frequency at which intermediate predictions are plotted
tolerance = 1e-8                      # Prescribed numerical tolerance (not explicitly used below)
learning_rate = 1e-2                  # Learning rate of the Adam optimizer

# Network architecture:
# [1, 10, 10, 10, 10, 1] means:
#   - One input neuron (state variable u)
#   - Four hidden layers, each containing 10 neurons
#   - One output neuron (predicted bifurcation parameter r)
layers = np.array([1, 10, 10, 10, 10, 1])


# ============================================================================
# SECTION 4: DEFINING THE PARAMETERS OF THE MATHEMATICAL MODEL
# ============================================================================

# Parameters appearing in the nonlinear dynamical system.
alpha = 0.1
beta = 2
h = 0.5
p = 2


# ============================================================================
# SECTION 5: DEFINING THE GOVERNING NONLINEAR SYSTEM
# ============================================================================

def G(p, h, u):
    """
    Define the nonlinear response/saturation function.

    Mathematical expression:
        G(p, h, u) = u^p / (u^p + h^p)

    Parameters:
        p : exponent controlling the shape of the response
        h : half-saturation parameter
        u : state variable

    Returns:
        Value of the nonlinear response function.
    """
    return u**p / (u**p + h**p)


def F(alpha, beta, r, p, h, u):
    """
    Define the right-hand side of the governing dynamical system.

    Mathematical model:
        du/dt = F(alpha, beta, r, p, h, u)
              = alpha - beta*u + r*G(p, h, u)

    At an equilibrium point u*, the time derivative is zero:
        F(alpha, beta, r, p, h, u*) = 0

    The neural network predicts r for a given input u. Its output is
    substituted into this equilibrium equation during training.

    Parameters:
        alpha, beta, p, h : model parameters
        r                 : neural-network-predicted bifurcation parameter
        u                 : state variable

    Returns:
        Value of the governing function F.
    """
    return alpha - beta * u + r * G(p, h, u)


# ============================================================================
# SECTION 6: IDENTIFYING LOCAL MAXIMA AND MINIMA
# ============================================================================

def Local_Max_Min(y, x):
    """
    Identify local extrema in a discretely sampled curve.

    A point is identified as a local extremum when its value is either
    greater than both neighbouring values or smaller than both
    neighbouring values.

    For three consecutive values y[i-1], y[i], y[i+1], the product

        (y[i] - y[i-1]) * (y[i] - y[i+1])

    is positive at a strict local maximum or minimum.

    Parameters:
        y : sampled function values
        x : corresponding independent-variable values

    Returns:
        local_x_values : x-coordinates of detected extrema
        local_values   : corresponding y-coordinates
    """

    # Compare each interior point with its immediate left and right neighbours.
    # The first and last points are excluded because they do not have
    # two neighbouring points.
    local_indices = (y[1:-1] - y[:-2]) * (y[1:-1] - y[2:]) > 0

    # Extract the function values at the detected local extrema.
    local_values = y[1:-1][local_indices]

    # Extract the corresponding input/state-variable values.
    local_x_values = x[1:-1][local_indices]

    return local_x_values, local_values


# ============================================================================
# SECTION 7: COMPUTING THE JACOBIAN MATRIX AND ITS EIGENVALUES
# ============================================================================

def Jacobian_and_eigenvalues(equations, variables, equilibrium_points):
    """
    Calculate the symbolic Jacobian matrix and its eigenvalues at
    specified equilibrium points.

    For a one-dimensional system:
        du/dt = F(u),

    the Jacobian is:
        J = dF/du.

    The sign of the Jacobian eigenvalue determines local stability:
        eigenvalue < 0  : locally asymptotically stable
        eigenvalue > 0  : unstable

    For a multidimensional system, all eigenvalues must be considered:
        all real parts negative : locally asymptotically stable
        at least one positive real part : unstable
        zero real part(s) : linearization may be inconclusive

    Parameters:
        equations          : list of symbolic governing equations
        variables          : list of symbolic state variables
        equilibrium_points : list of equilibrium-point coordinates

    Returns:
        Dictionary containing the Jacobian matrix and eigenvalues
        evaluated at each supplied point.
    """

    results = {}

    # Construct the Jacobian matrix.
    # Each entry is the partial derivative of an equation with respect
    # to one of the state variables.
    jacobian_matrix = sp.Matrix(
        [[sp.diff(eq, var) for var in variables] for eq in equations]
    )

    # Evaluate the Jacobian at every supplied equilibrium point.
    for eq_point in equilibrium_points:
        point_results = {}

        # Ensure that the equilibrium point is represented as a list.
        # This allows the same substitution procedure to be used for
        # both scalar and multidimensional equilibrium points.
        if not isinstance(eq_point, (list, tuple)):
            eq_point = [eq_point]

        # Substitute the equilibrium-point coordinates into the Jacobian.
        substituted_Jacobian = jacobian_matrix.subs(
            dict(zip(variables, eq_point))
        )
        point_results['Jacobian'] = substituted_Jacobian

        # Compute the eigenvalues of the evaluated Jacobian matrix.
        eigenvalues = substituted_Jacobian.eigenvals()
        point_results['eigenvalues'] = eigenvalues

        # Convert the equilibrium point to a tuple so it can be used
        # as a dictionary key (lists are not hashable).
        results[tuple(eq_point)] = point_results

    return results


# ============================================================================
# SECTION 8: CONSTRUCTING THE DEEP NEURAL NETWORK (DNN)
# ============================================================================

class DNN(nn.Module):
    """
    Define a fully connected feed-forward deep neural network.

    The network takes the state variable u as input and returns the
    predicted bifurcation parameter r as output.

    The architecture is specified by the 'layers' array.
    Tanh is used as the activation function in the hidden layers.
    """

    def __init__(self, layers):
        # Initialize the parent PyTorch neural-network class.
        super().__init__()

        # Select the hyperbolic tangent activation function.
        # Tanh introduces nonlinearity into the hidden layers.
        self.activation = nn.Tanh()

        # Create the fully connected linear layers.
        # Each layer connects the neurons of one layer to the next.
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])

        # Initialize the network weights using Xavier normal initialization.
        # This helps maintain a suitable scale of activations during training.
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)

            # Initialize all bias parameters to zero.
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, t):
        """
        Perform the forward propagation through the neural network.

        Input:
            t : tensor containing the state-variable values

        Output:
            Neural-network prediction of the bifurcation parameter r.
        """

        # Convert the input tensor to floating-point format.
        a = t.float()

        # Pass the input through all hidden layers.
        # A linear transformation followed by Tanh activation is applied
        # at each hidden layer.
        for i in range(len(layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)

        # Apply the final linear layer to obtain the network output.
        # No activation function is applied to the output layer.
        a = self.linears[-1](a)

        return a


# ============================================================================
# SECTION 9: DEFINING THE LOSS FUNCTION AND TRAINING MODEL
# ============================================================================

class FCN:
    """
    Define the model used to train the neural network.

    The loss function is constructed from the governing equilibrium
    equation. The network is trained to make the residual of that
    equation as small as possible.
    """

    def __init__(self, layers):

        # Mean squared error (MSE) measures the difference between
        # the governing-equation residual and zero.
        self.loss_function = nn.MSELoss()

        # Create the deep neural network with the prescribed architecture.
        self.dnn = DNN(layers)

    def loss_data(self, u):
        """
        Calculate the equilibrium residual loss.

        The neural network predicts r for each input u.
        The predicted value is substituted into

            F(alpha, beta, r, p, h, u) = 0.

        The resulting residual is compared with zero using MSE.

        A small residual indicates that the predicted pair (u, r)
        approximately satisfies the equilibrium equation.
        """

        # Make a copy of the input tensor.
        uc = u.clone()

        # Gradients with respect to the input are not required for this loss.
        uc.requires_grad = False

        # Define the target residual: zero at equilibrium.
        r_hat = torch.zeros(uc.shape[0]).reshape(-1, 1)

        # Obtain the predicted bifurcation parameter from the neural network.
        # The output is reshaped into a column vector.
        r = self.dnn(uc)[:, 0].reshape(-1, 1)

        # Substitute the predicted r and input u into the governing equation.
        # The MSE loss penalizes deviations of F from zero.
        loss_r = self.loss_function(
            F(alpha, beta, r, p, h, uc),
            r_hat
        )

        return loss_r

    def loss(self, u):
        """
        Return the total training loss.

        In this implementation, the total loss consists only of the
        equilibrium-equation residual loss.
        """

        loss_r = self.loss_data(u)
        loss_val = loss_r

        return loss_val


# ============================================================================
# SECTION 10: INITIALIZING THE NETWORK AND OPTIMIZER
# ============================================================================

# Create the neural-network-based model using the FCN class.
PINN = FCN(layers)

# Collect all trainable parameters (weights and biases) of the DNN.
params = list(PINN.dnn.parameters())

# Select the Adam optimization algorithm.
# Adam updates the network parameters to minimize the residual loss.
optimizer = torch.optim.Adam(params, lr=learning_rate)


# ============================================================================
# SECTION 11: TRAINING THE NEURAL NETWORK
# ============================================================================

# Lists for recording the training iteration and corresponding loss.
iteration = []
loss_value = []

# Begin the training process.
# The loop performs num_iter + 1 optimization steps.
for epoch in range(num_iter + 1):

    # Evaluate the loss over all training input points.
    loss = PINN.loss(u_train)

    # Clear gradients accumulated during the previous optimization step.
    optimizer.zero_grad()

    # Compute gradients of the loss with respect to network parameters.
    loss.backward()

    # Update the weights and biases using the Adam optimizer.
    optimizer.step()

    # Store the iteration number and loss value.
    # No gradient tracking is needed for recording these quantities.
    with torch.no_grad():
        iteration.append(epoch)
        loss_value.append(loss)

    # Display the current network prediction at prescribed intervals.
    if epoch % interval_iter == 0:
        with torch.no_grad():

            # Evaluate the network over the complete training domain.
            r1 = PINN.dnn(u_train)

            # Plot the predicted relationship between u and r.
            plt.plot(u_train, r1, color="green")
            plt.ylim((0, max(r1) + 0.1))
            plt.xlabel("State variable $u$")
            plt.ylabel("Predicted parameter $r$")
            plt.title(f"Network prediction at iteration {epoch}")
            plt.grid(True)
            plt.show()
            plt.pause(0.1)


# ============================================================================
# SECTION 12: PREPARING SYMBOLIC VARIABLES AND RESULT ARRAYS
# ============================================================================

# Define a symbolic state variable for the SymPy differentiation procedure.
x = sp.symbols('x')

# Arrays for storing the computed parameter and state-variable values.
# Stable arrays contain points classified as stable.
# Unstable arrays contain points classified as unstable.
t_rf = []                              # Reserved list; not used elsewhere in this script

t_stable_r = []
t_stable_us = []

t_unstable_r = []
t_unstable_us = []


# ============================================================================
# SECTION 13: EVALUATING THE NETWORK AND IDENTIFYING LOCAL EXTREMA
# ============================================================================

with torch.no_grad():

    # Evaluate the trained network at every training point.
    # The output represents the predicted bifurcation parameter r(u).
    t_r = PINN.dnn(u_train).reshape(-1, 1)

    # Identify local maxima and minima in the predicted r(u) curve.
    # These points are candidate turning points of the bifurcation curve.
    yp, xp = Local_Max_Min(t_r, u_train)

    # Plot the predicted bifurcation relationship.
    plt.plot(t_r, u_train, color="blue", label="Predicted equilibrium curve")

    # Set the horizontal-axis range.
    plt.xlim((0, max(t_r) + 0.1))

    # Mark the detected local extrema using red circles.
    plt.plot(xp, yp, 'ro', label="Detected local extrema")

    plt.xlabel("Bifurcation parameter $r$")
    plt.ylabel("State variable $u$")
    plt.title("Predicted equilibrium curve and local extrema")
    plt.grid(True)
    plt.legend()
    plt.show()

# Store the complete network prediction for use in the stability analysis.
original_t_r = PINN.dnn(u_train).reshape(-1, 1)


# ============================================================================
# SECTION 14: COMPUTING THE JACOBIAN EIGENVALUE AT EACH EQUILIBRIUM POINT
# ============================================================================

# Examine each input state-variable value and its corresponding predicted r.
for i in range(len(u_train)):

    # Obtain the current state-variable value.
    # A list is used because the Jacobian function expects a collection
    # of equilibrium-point coordinates.
    equilibria = [u_train[i].item()]

    # Extract the network-predicted bifurcation parameter at this point.
    t_r = original_t_r[i].item()

    # Construct the symbolic governing equation using the current
    # predicted value of r and the symbolic state variable x.
    eq1 = F(alpha, beta, t_r, p, h, x)

    # Store the governing equation and the corresponding symbolic variable.
    equations = [eq1]
    variables = [x]

    # Evaluate the Jacobian and its eigenvalues at the current state value.
    results = Jacobian_and_eigenvalues(
        equations,
        variables,
        equilibria
    )

    # Read the eigenvalue and classify the equilibrium point.
    for eq_point, data in results.items():

        # Extract the eigenvalues from the results dictionary.
        evs = data['eigenvalues']

        # Convert the real parts of the eigenvalues into a NumPy array.
        # For this one-dimensional system, there is one eigenvalue.
        ev_array = np.array([
            eigenvalue.as_real_imag()[0]
            for eigenvalue, multiplicity in evs.items()
        ])

        # A negative eigenvalue indicates local asymptotic stability
        # for this one-dimensional autonomous system.
        if ev_array[0] < 0:

            # Retain only points for which the predicted parameter r is nonnegative.
            if t_r >= 0:
                t_stable_r.append(t_r)
                t_stable_us.append(u_train[i].item())

        # A positive eigenvalue indicates instability.
        else:

            # Retain only points for which the predicted parameter r is nonnegative.
            if t_r >= 0:
                t_unstable_r.append(t_r)
                t_unstable_us.append(u_train[i].item())


# ============================================================================
# SECTION 15: PLOTTING THE STABLE AND UNSTABLE EQUILIBRIUM BRANCHES
# ============================================================================

# Plot the unstable equilibrium branch in red.
plt.plot(
    t_unstable_r,
    t_unstable_us,
    color="red",
    label="Unstable equilibrium branch"
)

# Plot the stable equilibrium branch in blue.
plt.plot(
    t_stable_r,
    t_stable_us,
    color="blue",
    label="Stable equilibrium branch"
)

# Add grid lines, axis labels, title, and legend.
plt.grid(True)
plt.xlabel("Bifurcation parameter $r$")
plt.ylabel("Equilibrium state $u_*$")
plt.title("Bifurcation diagram with stability classification")
plt.legend()
plt.show()


# ============================================================================
# SECTION 16: CONVERTING RESULTS TO NUMPY ARRAYS
# ============================================================================

# Convert the lists of stable and unstable points into NumPy arrays.
# This makes the results convenient for numerical analysis and export.
t_stable_r = np.array(t_stable_r)
t_stable_us = np.array(t_stable_us)

t_unstable_r = np.array(t_unstable_r)
t_unstable_us = np.array(t_unstable_us)

# Convert the training input tensor into a NumPy array.
u_train = np.array(u_train)

# Detach the network prediction from the PyTorch computation graph
# and convert it into a NumPy array.
original_t_r = original_t_r.detach().numpy()


# ============================================================================
# SECTION 17: EXPORTING THE COMPUTED RESULTS
# ============================================================================

# Save the bifurcation parameter values corresponding to stable points.
np.savetxt("t_stable_r.txt", t_stable_r)

# Save the state-variable values corresponding to stable points.
np.savetxt("t_stable_us.txt", t_stable_us)

# Save the bifurcation parameter values corresponding to unstable points.
np.savetxt("t_unstable_r.txt", t_unstable_r)

# Save the state-variable values corresponding to unstable points.
np.savetxt("t_unstable_us.txt", t_unstable_us)

# Save the original training input values.
np.savetxt("u_train.txt", u_train)

# Save the neural-network-predicted parameter values over the training domain.
np.savetxt("original_t_r.txt", original_t_r)

# The saved text files can be imported into MATLAB, Python, or other
# numerical software for further analysis and preparation of figures.


# ============================================================================
# END OF PROGRAM
# ============================================================================
