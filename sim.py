import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ====================
# 参数设置与初始化
# ====================
def add_perturbation(u, v, scale = 0.1, amplitude=1e-3):
    # 生成与速度场同尺寸的高斯噪声，均值为0，标准差为amplitude
    noise_u = amplitude * cp.random.normal(0, scale, u.shape)
    noise_v = amplitude * cp.random.normal(0, scale, v.shape)
    
    # 添加微扰至初始速度场
    u += noise_u
    v += noise_v

    u[:, :, 0] = 0    # 下边界
    u[:, :, -1] = 0   # 上边界
    v[:, :, 0] = 0
    v[:, :, -1] = 0
    return u, v

def initialize_parameters(max_steps=100):
    # 模拟域尺寸参数
    Lx = 3.0
    Ly = 3.0
    Lz = 1.0

    nx, ny, nz = 128, 128, 128   # 网格分辨率

    Ra = 2000                    # Rayleigh数
    Pr = 6.8                      # Prandtl数
    dt = 1e-6                    # 时间步长
    
    # 计算各方向空间步长
    dx_x = Lx / (nx-1)
    dx_y = Ly / (ny-1)
    dx_z = Lz / (nz-1)
    
    # 物理参数计算
    nu = cp.sqrt(Pr/Ra)
    kappa = 1/(Pr*nu)

    # 初始化场
    u = cp.zeros((nx, ny, nz))
    v = cp.zeros((nx, ny, nz))
    u, v = add_perturbation(u, v, amplitude=1e-5)
    w = cp.zeros((nx, ny, nz))
    T = cp.zeros((nx, ny, nz))
    p = cp.zeros((nx, ny, nz)) + Ra/Pr * (0.5 - cp.linspace(0,1,nz)[None,None,:])

    # 温度边界条件
    T[:, :, 0] = 1.0
    T[:, :, -1] = 0.0

    return nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, max_steps, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p

# ====================
# 辅助函数
# ====================
def gradient(f, dx_x, dx_y, dx_z):
    grad_x = cp.gradient(f, dx_x, axis=0)
    grad_y = cp.gradient(f, dx_y, axis=1)
    grad_z = cp.gradient(f, dx_z, axis=2)
    return grad_x, grad_y, grad_z

def laplacian(f, dx_x, dx_y, dx_z):
    lap_x = (cp.roll(f,1,0) + cp.roll(f,-1,0) - 2*f) / dx_x**2
    lap_y = (cp.roll(f,1,1) + cp.roll(f,-1,1) - 2*f) / dx_y**2
    lap_z = (cp.roll(f,1,2) + cp.roll(f,-1,2) - 2*f) / dx_z**2
    return lap_x + lap_y + lap_z

def advection(u, v, w, f, dx_x, dx_y, dx_z):
    grad_x, grad_y, grad_z = gradient(f, dx_x, dx_y, dx_z)
    return u*grad_x + v*grad_y + w*grad_z

def pressure_poisson(p, u_star, v_star, w_star, dx_x, dx_y, dx_z, dt, nx, ny, nz):

    # 使用FFT加速泊松方程求解
    rhs = (cp.gradient(u_star, dx_x, axis=0) + 
           cp.gradient(v_star, dx_y, axis=1) + 
           cp.gradient(w_star, dx_z, axis=2)) / dt

    
    # 三维FFT求解
    kx = cp.fft.fftfreq(nx, dx_x)[:, None, None]
    ky = cp.fft.fftfreq(ny, dx_y)[None, :, None]
    kz = cp.fft.fftfreq(nz, dx_z)[None, None, :]
    
    rhs_hat = cp.fft.fftn(rhs)
    inv_k2 = 1.0/(4*cp.pi**2*(kx**2 + ky**2 + kz**2))
    inv_k2[0,0,0] = 0.01  # 避免除以零
    
    p_hat = -inv_k2 * rhs_hat
    return cp.fft.ifftn(p_hat).real

# ====================
# 物理过程更新函数
# ====================
def update_velocity(u, v, w, p, u_star, v_star, w_star, grad_p, dt):

    # 校正步：更新速度场
    u = u_star - dt*grad_p[0]
    v = v_star - dt*grad_p[1]
    w = w_star - dt*grad_p[2]
    return u, v, w

def update_temperature(T, kappa, laplacian_T, advection_T, dt):
    # 温度场更新
    T += dt*(kappa*laplacian_T - advection_T)
    return T

# ====================
# 主计算函数
# ====================
def calc(nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, max_steps, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p ):
    for step in range(max_steps):
        if step % 500 == 0 and step > 0 :
            print(f"{step} step completed...")
        if step % 2000 == 0 and step > 1000:
            visualize_results(u, v, T, nx, ny, Lx, Ly, step)
            print("save results!")

         # 预测步
        u_star = u + dt*(nu*laplacian(u, dx_x, dx_y, dx_z) - advection(u,v,w,u, dx_x, dx_y, dx_z))
        v_star = v + dt*(nu*laplacian(v, dx_x, dx_y, dx_z) - advection(u,v,w,v, dx_x, dx_y, dx_z))
        w_star = w + dt*(nu*laplacian(w, dx_x, dx_y, dx_z) - advection(u,v,w,w, dx_x, dx_y, dx_z) + (Ra/Pr)*T)

        # print("u_star",u_star[32,32,32])

        # 压力泊松方程求解
        p = pressure_poisson(p, u_star, v_star, w_star, dx_x, dx_y, dx_z, dt, nx, ny, nz)
        grad_p = gradient(p, dx_x, dx_y, dx_z)

        # 更新速度场
        u, v, w = update_velocity(u, v, w, p, u_star, v_star, w_star, grad_p, dt)
        
        # 更新温度场
        laplacian_T = laplacian(T, dx_x, dx_y, dx_z)
        advection_T = advection(u, v, w, T, dx_x, dx_y, dx_z)
        T = update_temperature(T, kappa, laplacian_T, advection_T, dt)
        
        # 边界条件处理
        u[:, :, 0] = v[:, :, 0] = w[:, :, 0] = 0  # 无滑移
        T[:, :, 0] = 1.0; T[:, :, -1] = 0.0  # 温度固定

    return u, v, T

# ====================
# 可视化函数
# ====================
def visualize_results(u, v, T, nx, ny, Lx, Ly, step):
    # 将GPU数据复制回CPU
    surface_u = cp.asnumpy(u[:, :, -1])
    surface_v = cp.asnumpy(v[:, :, -1])
    surface_T = cp.asnumpy(T[:, :, -1])

    # 绘制温度场和流线
    plt.figure(figsize=(12,8))
    X, Y = np.mgrid[0:Lx:nx*1j, 0:Ly:ny*1j]

    plt.contourf(X, Y, surface_T.T, cmap='inferno', levels=25, alpha=0.6)
    # plt.colorbar(contour, label='Temperature (Normalized)')
    # plt.contourf(X, Y, surface_T.T, cmap='hot', levels=20)
    # plt.colorbar(label='Temperature')

    plt.streamplot(X.T, Y.T, surface_u.T, surface_v.T, color="darkblue", 
                           cmap='winter', linewidth=2.5, density=1.2,
                            minlength=0.1, maxlength=2.0)
    # plt.colorbar(stream.lines, label='Flow Speed (m/s)')
    # plt.streamplot(X.T, Y.T, surface_u.T, surface_v.T, 
    #                color='white', density=2, linewidth=1)
    
    plt.title(f'Rayleigh-Bénard Convection Surface Pattern (Ra={Ra})')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.savefig(f"/data/group_003/computer_physics/rb/Timestep{step}") 
# ====================
# 主程序
# ====================
if __name__ == "__main__":
    print("Start simulating...")

    # 参数初始化
    nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, max_steps, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p = initialize_parameters(max_steps=20000)

    # 主计算
    u, v, T = calc(nx, ny, nz, Lx, Ly, Lz, Ra, Pr, dt, max_steps, dx_x, dx_y, dx_z, nu, kappa, u, v, w, T, p )

    # 可视化结果
    visualize_results(u, v, T, nx, ny, Lx, Ly, max_steps)