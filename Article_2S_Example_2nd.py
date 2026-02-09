import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

# create
start_u = 0.05
end_u = 0.75
start_v = 0.01
end_v = 0.21
steps = 1401

u_train = torch.linspace(start_u, end_u, steps).unsqueeze(dim=1)
v_train = torch.linspace(start_v, end_v, steps).unsqueeze(dim=1)

# Parameter values
alpha = 0.02
beta = 2.0


num_iter = 500000
interval_iter = 100
tolerance = 1e-15
learning_rate = 5e-3

layers = np.array([1, 10, 10, 10, 10, 1])  # 3 hidden layers


# System
def F1(u, v):
    return u * (1 - u) - u * v / (u ** 2 + alpha)


def F2(u, v, gam):
    return v * (gam - beta * v / u)


def Jacobian_and_eigenvalues(equations, variables, equilibrium_points):
    results = {}
    jacobian_matrix = sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])

    for eq_point in equilibrium_points:
        point_results = {}

        # Substitute the equilibrium point values into the Jacobian matrix.
        substituted_Jacobian = jacobian_matrix.subs(dict(zip(variables, eq_point)))
        point_results['Jacobian'] = substituted_Jacobian

        # Calculate eigenvalues of the substituted Jacobian.
        eigenvalues = substituted_Jacobian.eigenvals()
        point_results['eigenvalues'] = eigenvalues
        results[eq_point] = point_results
    return results


# print(X_test, Y_test)
# len(X_train)


def plot(r_pts=None, u_prediction=None, i=None, xp=None, yp=None):
    # plt.figure(figsize=(12, 4))
    plt.clf()
    if r_pts is not None:
        plt.plot(r_pts, u_prediction, label="Prediction at iter {}".format(i), color="green")
        plt.legend()
        plt.xlabel('gamma')
        plt.ylabel('u_{*}')

    if xp is not None:
        plt.plot(yp, xp, 'ro')


class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()  # call __init__ from parent class
        'activation function'
        self.activation = nn.Tanh()
        'Initialize neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        'Xavier Normal Initialization'
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    'foward pass'

    def forward(self, t):
        # convert to float
        a = t.float()
        for i in range(len(layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)
        # a = 0.3 + torch.sigmoid(a)*0.4
        # a = 2.5 + a*0.7
        return a


class FCN():
    def __init__(self, layers):
        self.loss_function = nn.MSELoss()

        'Call our DNN'
        self.dnn = DNN(layers)

    'loss function'

    def loss_data(self, t):
        # Initialize loss_f1 and loss_g1 directly with the desired shape
        loss_f1 = torch.zeros(len(u_train), dtype=t.dtype, device=t.device)

        # Initialize AF1 with the same shape and type as F1
        AF1 = torch.zeros(len(u_train), dtype=t.dtype, device=t.device)

        T = t.clone()
        T.requires_grad = False
        u_hat = torch.zeros(len(u_train), 1, dtype=t.dtype, device=t.device)
        t_u = T.reshape(-1, 1)
        # Calculate t_a2 outside the no_grad context
        t_r = self.dnn(T).reshape(-1, 1)  #This ensures t_a2 has gradient information

        for n in range(len(u_train)):
            # Create new tensors for f1 and g1 in each iteration
            # Call the F1 and F2 functions defined outside the class
            f1 = F1(t_u[n], v_train)
            g1 = F2(t_u[n], v_train, t_r[n])

            loss_f1[n] = torch.min(f1 ** 2 + g1 ** 2)  # Use loss_f1 to store the loss value
            AF1[n] = v_train[torch.argmin(f1 ** 2 + g1 ** 2)]

        loss_u = self.loss_function(loss_f1.reshape(-1, 1), u_hat)  # Use loss_f1 in loss calculation

        return loss_u, AF1

    def loss(self, t):
        loss_u, vs = self.loss_data(t)

        loss_val = loss_u
        return loss_val, vs


def Local_Max_Min(y, x):
    # Find local minima
    local_indices = (y[1:-1] - y[:-2]) * (y[1:-1] - y[2:]) > 0
    local_values = y[1:-1][local_indices]
    local_x_values = x[1:-1][local_indices]

    return local_x_values, local_values



EINN = FCN(layers)
params = list(EINN.dnn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

iteration = []
loss_value = []
df_theta = []
for epoch in range(num_iter + 1):
    loss, v_s = EINN.loss(u_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss.item() < tolerance:
        break
    if epoch % interval_iter == 0:
        with torch.no_grad():
            v_eq = v_s
            t_r2 = EINN.dnn(u_train)
            # modified = str(t_r2).replace('[', '').replace(']', '').replace(',', '')
            print(loss.item())
            df_theta.append(t_r2)
            # df_theta.append(modified)

            xp, yp = Local_Max_Min(t_r2, u_train)
            if yp is not None:
                yp = yp.numpy()
                xp = xp.numpy()
            if yp is not None:
                plot(r_pts=t_r2, u_prediction=u_train, i=epoch, xp=xp, yp=yp)
            else:
                plot(r_pts=t_r2, u_prediction=u_train, i=epoch, xp=None, yp=None)
            plt.pause(0.01)

# theta_df = pd.DataFrame(df_theta)

# np.savetxt('Prediction_out_a2.txt', df_theta)

df_theta_np = np.array(df_theta)
# Reshape the array from (epochs, data_points, output_dim) to (epochs, data_points)
# Since the output_dim is 1, we can simply squeeze the last dimension.
df_theta_np = df_theta_np.squeeze(axis=-1)
np.savetxt("A_2S_2nd_df_theta.txt", df_theta_np)

np.savetxt("A_2S_2nd_u_train.txt", u_train)
np.savetxt("A_2S_2nd_v_train.txt", v_train)
np.savetxt("A_2S_2nd_gam_eq.txt", t_r2)
np.savetxt("A_2S_2nd_v_eq.txt", v_eq)