import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Parameter Initialization
def add_perturbation(u, v, scale=0.1, amplitude=1e-3):
    noise_u = amplitude * cp.random.normal(0, scale, u.shape)
    noise_v = amplitude * cp.random.normal(0, scale, v.shape)
    u += noise_u
    v += noise_v
    u[:, :, 0] = 0; u[:, :, -1] = 0
    v[:, :, 0] = 0; v[:, :, -1] = 0
    return u, v

def initialize_parameters(max_steps=100):
    Lx, Ly, Lz = 0.1, 0.1, 0.004
    nx, ny, nz = 64, 64, 64
    Ra, Pr, dt = 10000, 6.8, 1e-8
    dx_x, dx_y, dx_z = Lx / (nx-1), Ly / (ny-1), Lz / (nz-1)
    nu = cp.sqrt(Pr / Ra)
    kappa = 1 / (Pr * nu)
    u = cp.zeros((nx, ny, nz))
    v = cp.zeros((nx, ny, nz))
    u, v = add_perturbation(u, v, amplitude=1e-5)
    w = cp.zeros((nx, ny, nz))
    T = cp.zeros((nx, ny, nz))
    p = cp.zeros((nx, ny, nz)) + Ra/Pr * (0.5 - cp.linspace(0, 1, nz)[None, None, :])
    T[:, :, 0] = 100; T[:, :, -1] = 0.0
    return nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, max_steps, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p

# Helper Functions
def gradient(f, dx_x, dx_y, dx_z):
    grad_x = cp.gradient(f, dx_x, axis=0)
    grad_y = cp.gradient(f, dx_y, axis=1)
    grad_z = cp.gradient(f, dx_z, axis=2)
    return grad_x, grad_y, grad_z

def laplacian(f, dx_x, dx_y, dx_z):
    lap_x = (cp.roll(f, 1, 0) + cp.roll(f, -1, 0) - 2*f) / dx_x**2
    lap_y = (cp.roll(f, 1, 1) + cp.roll(f, -1, 1) - 2*f) / dx_y**2
    lap_z = (cp.roll(f, 1, 2) + cp.roll(f, -1, 2) - 2*f) / dx_z**2
    return lap_x + lap_y + lap_z

def advection(u, v, w, f, dx_x, dx_y, dx_z):
    grad_x, grad_y, grad_z = gradient(f, dx_x, dx_y, dx_z)
    return u*grad_x + v*grad_y + w*grad_z

def pressure_poisson(u_star, v_star, w_star, dx_x, dx_y, dx_z, dt, nx, ny, nz, Lx, Ly):
    grad_u_x = cp.gradient(u_star, dx_x, axis=0)
    grad_v_y = cp.gradient(v_star, dx_y, axis=1)
    grad_w_z = cp.gradient(w_star, dx_z, axis=2)
    rhs = (grad_u_x + grad_v_y + grad_w_z) / dt
    rhs_hat = cp.fft.fft2(rhs, axes=(0, 1))
    w_star_hat = cp.fft.fft2(w_star, axes=(0, 1))
    kx = 2 * cp.pi * cp.fft.fftfreq(nx, dx_x)
    ky = 2 * cp.pi * cp.fft.fftfreq(ny, dx_y)
    p_hat = cp.zeros((nx, ny, nz), dtype=cp.complex128)
    for i in range(nx):
        for j in range(ny):
            k2 = kx[i]**2 + ky[j]**2
            A = cp.zeros((nz, nz), dtype=cp.complex128)
            for k in range(1, nz-1):
                A[k, k-1] = 1 / dx_z**2
                A[k, k] = -2 / dx_z**2 - k2
                A[k, k+1] = 1 / dx_z**2
            A[0, 0] = -1 / dx_z; A[0, 1] = 1 / dx_z
            A[-1, -2] = -1 / dx_z; A[-1, -1] = 1 / dx_z
            b = cp.zeros(nz, dtype=cp.complex128)
            b[0] = w_star_hat[i, j, 0] / dt
            b[1:-1] = rhs_hat[i, j, 1:-1]
            b[-1] = w_star_hat[i, j, -1] / dt
            if i == 0 and j == 0:
                A[0, :] = 0; A[0, 0] = 1; b[0] = 0
            p_hat[i, j, :] = cp.linalg.solve(A, b)
    p = cp.fft.ifft2(p_hat, axes=(0, 1)).real
    return p - p.mean()

def update_velocity(u, v, w, p, u_star, v_star, w_star, grad_p, dt):
    u = u_star - dt * grad_p[0]
    v = v_star - dt * grad_p[1]
    w = w_star - dt * grad_p[2]
    return u, v, w

def update_temperature(T, kappa, laplacian_T, advection_T, dt):
    T += dt * (kappa * laplacian_T - advection_T)
    return T

# Main Calculation
def calc(nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, max_steps, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p):
    for step in range(max_steps):
        if step % 10 == 0 and step > 0:
            print(f"{step} step completed...")
        if step % 20 == 0 and step > 0:
            visualize_results(nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, step, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p)
            print("save results!")
        u_star = u + dt * (nu * laplacian(u, dx_x, dx_y, dx_z) - advection(u, v, w, u, dx_x, dx_y, dx_z))
        v_star = v + dt * (nu * laplacian(v, dx_x, dx_y, dx_z) - advection(u, v, w, v, dx_x, dx_y, dx_z))
        w_star = w + dt * (nu * laplacian(w, dx_x, dx_y, dx_z) - advection(u, v, w, w, dx_x, dx_y, dx_z) + (Ra/Pr) * T)
        p = pressure_poisson(u_star, v_star, w_star, dx_x, dx_y, dx_z, dt, nx, ny, nz, Lx, Ly)
        grad_p = gradient(p, dx_x, dx_y, dx_z)
        u, v, w = update_velocity(u, v, w, p, u_star, v_star, w_star, grad_p, dt)
        u[:, :, 0] = 0
        v[:, :, 0] = 0
        w[:, :, 0] = 0
        laplacian_T = laplacian(T, dx_x, dx_y, dx_z)
        advection_T = advection(u, v, w, T, dx_x, dx_y, dx_z)
        T = update_temperature(T, kappa, laplacian_T, advection_T, dt)
        T[:, :, 0] = 100; T[:, :, -1] = 0.0
    return u, v, T

# Visualization
def visualize_results(nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, step, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p):
    pos = -1  # Middle plane for better visualization
    surface_u = cp.asnumpy(u[:, :, pos])
    surface_v = cp.asnumpy(v[:, :, pos])
    surface_T = cp.asnumpy(T[:, :, pos])
    plt.figure(figsize=(12, 8))
    X, Y = np.mgrid[0:Lx:nx*1j, 0:Ly:ny*1j]
    plt.contourf(X, Y, surface_T, cmap='inferno', levels=25, alpha=0.6)
    plt.streamplot(X.T, Y.T, surface_u.T, surface_v.T, color="darkblue",
                   cmap='winter', linewidth=2.5, density=1.2,
                   minlength=0.1, maxlength=2.0)
    plt.title(f'Rayleigh-Bénard Convection at z={pos*dx_z:.2f} (Ra={Ra},Pr={Pr})')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.savefig(f"/data/group_003/computer_physics/rb2/Timestep{step}")
    plt.close()

# Main Program
if __name__ == "__main__":
    device_id = 2
    with cp.cuda.Device(device_id):
        print("Start simulating...")
        params = initialize_parameters(max_steps=20000)
        u, v, T = calc(*params)
        visualize_results(*params)
