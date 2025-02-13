import numpy as np
from scipy.integrate import solve_ivp

# Fourth-order Runge-Kutta scheme from
# https://github.com/MaxRamgraber/Triangular-Transport-Toolbox/blob/26155c22279b727ded570675689e7ed69703daf6/Examples%20C%20-%20data%20assimilation/Example%2005%20-%20Ensemble%20Transport%20Filter/example_05.py#L49
def rk4(Z, fun, t1, nt):
    """
    In-place fourth-order Runge-Kutta integration.
    Parameters
        Z       : initial states
        fun     : function to be integrated
        t       : final time
        nt      : number of time steps

    """

    # Go through all time steps
    Z_tmp = np.empty_like(Z)
    dt = t1 / nt
    for i in range(nt):
        # Calculate the RK4 values
        k1 = fun(i * dt, Z)
        Z_tmp[:] = Z + (dt / 2) * k1
        k2 = fun(i * dt + 0.5 * dt, Z_tmp)
        Z_tmp[:] = Z + (dt / 2) * k2
        k3 = fun(i * dt + 0.5 * dt, Z_tmp)
        Z_tmp[:] = Z + dt * k3
        k4 = fun(i * dt + dt, Z_tmp)
        # Update next value
        Z[:] += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(Z, fun, t1, nt):
    """
    In-place Euler integration for t in (0,t1)

    Parameters
        Z       : initial states
        fun     : function to be integrated
        t1      : final time
        nt      : number of time steps

    """
    dt = t1 / nt
    # Go through all time steps
    for i in range(nt):
        # Calculate the Euler values
        Z[:] += dt * fun(i * dt, Z)

def scipy_int(Z, fun, t1, nt, method):
    """
    inplace scipy integration
    """
    sol = solve_ivp(fun, (0, t1), Z, t_eval=(t1,), method=method)
    # print(f'sol.shape {sol.y.shape} Z.shape {Z.shape}')
    Z[:] = sol.y.reshape(-1)


class GradientFlowIntegrator:
    def __init__(self, method="RK4"):
        self.method = method
        if method == "RK4":
            self.integrator = rk4
        elif method == "Euler":
            self.integrator = euler
        else:
            self.integrator = lambda *args: scipy_int(*args, method = method)

    def __call__(self, Z, fun, t1, nt):
        self.integrator(Z, fun, t1, nt)
