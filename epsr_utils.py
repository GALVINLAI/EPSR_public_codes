import numpy as np
from epsr import EPSR
from scipy.optimize import differential_evolution, basinhopping, shgo

def L1normFinalVar(x, Omegas, d):
    b=EPSR(x, Omegas, d)    
    return np.sum(np.abs(b))

def L2norm2FinalVar(x, Omegas, d):
    b=EPSR(x, Omegas, d)    
    return 0.5*np.sum(np.abs(b)**2)


# for d = 1
def numerical_optimal_nodes(Omegas, shot_scheme):
    d=1
    if shot_scheme == 'weighted':
        cost = lambda x: L1normFinalVar(x, Omegas, d)
    elif shot_scheme == 'uniform':
        cost = lambda x: L2norm2FinalVar(x, Omegas, d)
    else:
        raise ValueError("Invalid shot_scheme. Please provide 'weighted' or 'uniform'.")

    r = len(Omegas)
    bounds = [(0.01, np.pi-0.01) for _ in range(r)]
    result = differential_evolution(cost, bounds)

    # Optimal x_mu and corresponding minimized l1 norm
    optimal_nodes = result.x
    optimal_nodes.sort()
    optimal_value = result.fun

    return optimal_nodes, optimal_value

# for d = 1
def compute_derivative_using_psr(estimate_loss_, weights, j, N_total, nodes, b, shot_scheme):
    def noise_univariate_fun(x, Nx_shots):
        return estimate_loss_(np.concatenate([weights[:j], [x], weights[j+1:]]), Nx_shots)

    gamma = 0.5 * np.hstack((b, -b)) # for odd d

    if shot_scheme == 'weighted':
        gamma_l1norm = np.sum(np.abs(gamma))
        Nx_list = [int(N_total * np.abs(gamma[i]) / gamma_l1norm) for i in range(len(gamma))]
    elif shot_scheme == 'uniform':
        Nx_list = [int(N_total / len(gamma)) for _ in range(len(gamma))]
    else:
        raise ValueError("Invalid shot_scheme. Please provide 'weighted' or 'uniform'.")

    phi = np.hstack((nodes, -nodes))
    x0 = weights[j]
    shifted_x0 = x0 + phi
    f_values = np.array([noise_univariate_fun(value, Nx) for value, Nx in zip(shifted_x0, Nx_list)])
    derivative = np.sum(gamma * f_values)

    if shot_scheme == 'weighted':
        total_var = np.sum(np.abs(b))**2
    elif shot_scheme == 'uniform':
        total_var = len(b) * np.sum(np.abs(b)**2)

    return derivative, b, total_var

def compute_derivative_using_psr_ture(expectation_loss, weights, j, N_total, nodes, b, shot_scheme):

    ture_univariate_fun = lambda x: expectation_loss(np.concatenate([weights[:j], [x], weights[j+1:]]))

    gamma = 0.5 * np.hstack((b, -b)) # for odd d

    phi = np.hstack((nodes, -nodes))
    x0 = weights[j]
    shifted_x0 = x0 + phi
    f_values = np.array([ture_univariate_fun(value) for value in shifted_x0])
    der = np.sum(gamma * f_values)
    return der, b

