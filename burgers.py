import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def burg_system(u,t,k,mu,nu):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j*k*u_hat
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -mu*u*u_x + nu*u_xx
    return u_t.real

def burgers(mu, nu, Lx, dx, Lt, dt):
    '''
    mu/nu: balance the non-linear and diffusion process.
    Lx: spatial domain length.
    dx: spatial domain resolution.
    Lt: temporal domain length.
    dt: temporal domain resolution.
    '''
    Nx = int(Lx/dx)
    Nt = int(Lt/dt)

    X = np.linspace(0,Lx,Nx) #Spatial array
    T = np.linspace(0,Lt,Nt) #Temporal array

    u0 = np.zeros(100) #np.exp(-(X-3)**2/2) # initial condition
    print("init condition shape", u0.shape)
    #Wave number discretization
    k = 2*np.pi*np.fft.fftfreq(Nx, d = dx)
    return X, T, u0, k, mu, nu

def vis_domain(X, T, U):
    xv, tv = np.meshgrid(X, T)
    fig, ax = plt.subplots(figsize=(15,5))
    ax.pcolormesh(xv, tv, U)
    plt.show()
    return



if __name__ == "__main__":
    X, T, u0, k, mu, nu = burgers(mu=1, nu=0.01, Lx=10, dx=.1, Lt=100, dt=0.025)
    U = odeint(burg_system, u0, T, args=(k, mu, nu,), mxstep=5000).T
    vis_domain(T, X, U)