import numpy as np
import torch
from math import pi, sqrt, sin, cos
from scipy.stats import norm
from pinn import pinn_func

# Structure parameters
L  = 9
m  = 250
EI = 4e7
w1 = pi**2 * sqrt(EI/(m*L**4))

wb = 0.6 * w1
A1 = 4e4
A2 = 6e4
phi1 = pi/4
phi2 = 3*pi/4

# Representative point
def h_w(s, n):
    x = np.zeros([n, s])
    if s == 2:
        q = np.arange(1, n + 1, 1)
        q = q[:, np.newaxis]
        a, b = 1, 1
        while n != b:
            a, b = b, a + b
        Q = np.array([[1, a]])
        x = (2*np.multiply(q, Q)-1)/(2*n) - np.trunc((2*np.multiply(q, Q)-1)/(2*n))
    return x

s, n_cases = 2, 34
theta = h_w(s, n_cases)
A = A1 + (A2-A1)*theta[:, 0]
phi = phi1 + (phi2-phi1)*theta[:, 1]
Pq = np.loadtxt('asgn_prob.txt')

# Physical and probability solutions
ymin, ymax = -0.1, 0.1
tmin, tmax = 1.0, 1.2
M, N = 200, 200
h, dt = (ymax-ymin)/M, (tmax-tmin)/N
yc = torch.linspace(ymin, ymax, M+1)
tc = np.linspace(tmin, tmax, N+1)
sigma = 0.0025

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selecting the device:", device)

for i in range(1):
    print(f"\n=== Case {i+1}/{n_cases} Start Training ===")
    y0 = 2*A[i]*L**3/(pi**4*EI) * w1 * (
        w1*(cos(w1*tc[0])*cos(phi[i]) - cos(wb*tc[0] + phi[i]))
        - wb*sin(w1*tc[0])*sin(phi[i])
    ) / (wb**2 - w1**2)
    print("Initial y0:", y0)

    p0 = Pq[i] / (norm.cdf(ymax, y0, sigma) - norm.cdf(ymin, y0, sigma))
    def p0y(y):
        return 1/(np.sqrt(2*np.pi)*sigma) * torch.exp(-(y-y0)**2/(2*sigma**2))
    def ypnt(t):
        return 2*A[i]*L**3/(pi**4*EI) * w1 * (
            w1*(torch.cos(w1*t)*cos(phi[i]) - torch.cos(wb*t + phi[i]))
            - wb*torch.sin(w1*t)*sin(phi[i])
        ) / (wb**2 - w1**2)
    def ydot(t):
        return 2*A[i]*L**3/(pi**4*EI) * w1 * (
            w1*w1*torch.sin(w1*t)*cos(phi[i]) - w1*wb*torch.sin(wb*t + phi[i])
            + wb*w1*torch.cos(w1*t)*sin(phi[i])
        ) / (w1**2 - wb**2)

    pyt = pinn_func(y0, h, ymin, ymax, tmin, tmax, p0, p0y, ypnt, ydot, idx=i, device=device)

    with torch.no_grad():
        t_grid, y_grid = np.meshgrid(tc, yc.cpu().numpy())
        t_flat = torch.tensor(t_grid.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
        y_flat = torch.tensor(y_grid.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
        yt = torch.cat([t_flat, y_flat], dim=1)
        p_pred = pyt(yt)
        p_pred = p_pred.detach().cpu().numpy().reshape(M+1, N+1)
    

    data = np.stack((t_grid.ravel(), y_grid.ravel(), p_pred.ravel()), axis=-1)
    filename = f'beam_exc_pinn_{i+1}.txt'
    np.savetxt(filename, data, fmt='%.8e', delimiter='\t', header='tc\tyc\tp')
    print(f"=== Case {i+1}/{n_cases} Training finished ===")

print("All done!")