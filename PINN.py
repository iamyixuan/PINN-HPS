from math import pi as PI
from math import exp as exp
import random
import numpy as np
import tensorflow as tf



def analytical_solution(NT, NX, TMAX, XMAX, NU):
   """
   Returns the velocity field and distance for the analytical solution
   """

   # Increments
   DT = TMAX/(NT-1)
   DX = XMAX/(NX-1)

   # Initialise data structures
   import numpy as np
   u_analytical = np.zeros((NX,NT))
   x = np.zeros(NX)
   t = np.zeros(NT)

   # Distance
   for i in range(0,NX):
       x[i] = i*DX

   # Analytical Solution
   for n in range(0,NT):
       t = n*DT

       for i in range(0,NX):
           phi = exp( -(x[i]-4*t)**2/(4*NU*(t+1)) ) + exp( -(x[i]-4*t-2*PI)**2/(4*NU*(t+1)) )

           dphi = ( -0.5*(x[i]-4*t)/(NU*(t+1))*exp( -(x[i]-4*t)**2/(4*NU*(t+1)) )
               -0.5*(x[i]-4*t-2*PI)/(NU*(t+1))*exp( -(x[i]-4*t-2*PI)**2/(4*NU*(t+1)) ) )

           u_analytical[i,n] = -2*NU*(dphi/phi) + 4

   return u_analytical, x


def convection_diffusion(NT, NX, TMAX, XMAX, NU):
    DT = TMAX/(NT-1)
    DX = XMAX/(NX-1)

    # Initialise data structures
    u = np.zeros((NX,NT))
    u_analytical = np.zeros((NX,NT))
    x = np.zeros(NX)
    t = np.zeros(NT)
    ipos = np.zeros(NX, dtype=int)
    ineg = np.zeros(NX, dtype=int)

   # Periodic boundary conditions
    for i in range(0,NX):
        x[i] = i*DX
        ipos[i] = i+1
        ineg[i] = i-1

    ipos[NX-1] = 0
    ineg[0] = NX-1

   # Initial conditions
    for i in range(0,NX):
        phi = exp( -(x[i]**2)/(4*NU) ) + exp( -(x[i]-2*PI)**2 / (4*NU) )
        dphi = -(0.5*x[i]/NU)*exp( -(x[i]**2) / (4*NU) ) - (0.5*(x[i]-2*PI) / NU )*exp(-(x[i]-2*PI)**2 / (4*NU) )
        u[i,0] = -2*NU*(dphi/phi) + 4
        #u[i, 0] = -np.sin(PI*x[i])
    
   # Numerical solution
    for n in range(0,NT-1):
        for i in range(0,NX):
            u[i,n+1] = (u[i,n]-u[i,n]*(DT/DX)*(u[i,n]-u[ineg[i],n])+ NU*(DT/DX**2)*(u[ipos[i],n]-2*u[i,n]+u[ineg[i],n]))

    return u, x



class PINN(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim, act_fn):
        super().__init__()
        self.lys = [tf.keras.layers.Dense(units=hidden_dim, activation=act_fn)]
        for n in range(num_layers):
            self.lys.append(tf.keras.layers.Dense(units=hidden_dim, activation=act_fn))
        self.lys.append(tf.keras.layers.Dense(units=output_dim, activation=None))
    
    def call(self, x):
        x = tf.cast(x, tf.float32)
        for layer in self.layers:
            x = layer(x)
        return x

class BurgerSupervisor:
    def __init__(self, nu, net, epochs, batch_size, lr, alpha):
        self.nu = nu
        self.net = net
        self.epochs = epochs
        self.batch_size = batch_size

        self.alpha = alpha
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def u_loss(self, u_true, x):
        u_l = tf.reduce_mean((self.net(x) - u_true)**2)
        return u_l 

    def f_loss(self, x):
        x = tf.cast(x, tf.float32)
        t = tf.Variable(x[:, 1])
        x = tf.Variable(x[:, 0])
        with tf.GradientTape() as t2:
            with tf.GradientTape(persistent=True) as tape:
                inputs = tf.stack((x,t), axis=1)
                u_ = self.net(inputs)
            u_x = tape.gradient(u_, x)
            u_t = tape.gradient(u_, t)
        u_xx = t2.gradient(u_x, x)
        f = u_t + (u_ * u_x) - (self.nu * u_xx)
        return tf.reduce_mean(f**2)
    
    def loss(self, x_u, y_u, x_f):
        total_loss = self.u_loss(y_u, x_u) + self.alpha*self.f_loss(x_f) 
        return total_loss
    
    def train(self, x_train, y_train, x_val, y_val):
        x_u_train = x_train[0]
        x_f_train = x_train[1]
        for i in range(self.epochs):
            train_ls = []
            for j in range(len(x_u_train) // self.batch_size):
                x_train_batch = x_u_train[j*self.batch_size : (j+1)*self.batch_size]
                y_train_batch = y_train[j*self.batch_size : (j+1)*self.batch_size]
                with tf.GradientTape() as t:
                    loss_ = self.loss(x_train_batch, y_train_batch, x_f_train)
                    train_ls.append(loss_)
                grads = t.gradient(loss_, self.net.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
            
            # validation loss
            val_r2 = self.r2(y_val, self.net(x_val))
            val_f_loss = self.f_loss(x_val)
            print("Epoch %d, training loss: %.4f, validation r2: %.4f, validiation PDE loss %.4f" % (i+1, tf.reduce_mean(train_ls), val_r2, val_f_loss))
        return val_r2, val_f_loss # return the objective.
                
    def r2(self, true, pred):
        true = tf.cast(true, tf.float32)
        rss = tf.reduce_sum((true - pred)**2)  
        tss = tf.reduce_sum((true - tf.reduce_mean(true))**2)
        return 1 - rss/tss

    def test(self, x_test, y_test):
        out = self.net(x_test)
        r2 = self.r2(y_test, out)
        f_loss = self.f_loss(x_test)
        return r2, f_loss


    def plot(self):
        pass


def get_data(NT, NX, TMAX, XMAX, NU):
    """
    NT: number of points in time.
    NX: number of points in space.
    TMAX: the maximum time span.
    XMAX: the maximum space span.
    NU: viscosity
    """
    u,x = convection_diffusion(NT, NX, TMAX, XMAX, NU) # shape of [x, t]
    x = np.linspace(0, XMAX, NX)
    t = np.linspace(0, TMAX, NT)
    xv, tv = np.meshgrid(x, t)
    x_= np.stack((xv.flatten(), tv.flatten()), axis=1)
    y_= u.T.flatten()

    # Get the indices of points on the boundaries and initial conditions.
    b_up_idx = np.where(x_[:,0]==x[-1])[0]
    b_low_idx = np.where(x_[:,0]==x[0])[0]
    ic_idx = np.where(x_[:,1]==t[0])[0]
  
    u_train_idx = np.concatenate([b_up_idx, b_low_idx, ic_idx]).reshape(-1)
    x_u_train = x_[u_train_idx]
    y_u_train = y_[u_train_idx]

    NUM_BC_PT = 200
    u_idx = np.random.permutation(len(x_u_train))[:NUM_BC_PT]
    NUM_COL_PT = 10000#int(.1 * len(x_)) # 10% of the points in domain.
    print("Generating data... total size %d, number of collocation points for trianing %d" % (len(x_), NUM_COL_PT))
    all_idx = np.arange(len(x_)).tolist()
    col_idx = [i for i in all_idx if (i not in u_train_idx)]
    f_idx = random.sample(col_idx, NUM_COL_PT)

    x_f_train = x_[f_idx]
    # y_f_train = y_[f_idx]

    x_test = x_[col_idx]
    y_test = y_[col_idx]
    return (x_u_train[u_idx], x_f_train), y_u_train[u_idx], (x_test, y_test)


if __name__ == "__main__":
    # import matplotlib.animation as animation
    # u_analytical,x = convection_diffusion(1024, 256, 2, 2.0*PI, 0.01)
    # print(u_analytical.shape)

    # x = np.linspace(0, 2.0*PI, 256)
    # t = np.linspace(0, 2, 1024)
    # xv, tv = np.meshgrid(x, t)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 1)
    # ax[0].pcolormesh(xv, tv, u_analytical.T, shading='nearest')
    # ax[0].set_xlabel('t')
    # ax[0].set_ylabel('x')


    # line,= ax[1].plot(x, u_analytical[:, 50])
    # text = ax[1].text(0.8, 0.9, ' ', transform=ax[1].transAxes)
    # ax[1].set_xlabel('x')
    # ax[1].set_ylabel('u')
    # def animate(t):
    #     line.set_ydata(u_analytical[:, t])
    #     text.set_text('t=%d'%t)
    #     return line, text
    
    # ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)
    # plt.tight_layout()
    # ani.save('./Burgers.gif', writer='imagemagick', fps=60)
    # plt.show()
    nu = 0.01 / np.pi         # constant in the diff. equation
    N_u = 100                 # number of data points in the boundaries
    N_f = 10000               # number of collocation points

    # X_u_train: a set of pairs (x, t) located at:
        # x =  1, t = [0,  1]
        # x = -1, t = [0,  1]
        # t =  0, x = [-1, 1]
    x_upper = np.ones((N_u//4, 1), dtype=float)
    print(x_upper)
    x_lower = np.ones((N_u//4, 1), dtype=float) * (-1)
    t_zero = np.zeros((N_u//2, 1), dtype=float)

    t_upper = np.random.rand(N_u//4, 1)
    t_lower = np.random.rand(N_u//4, 1)
    x_zero = (-1) + np.random.rand(N_u//2, 1) * (1 - (-1))

    # stack uppers, lowers and zeros:
    X_upper = np.hstack( (x_upper, t_upper) )
    X_lower = np.hstack( (x_lower, t_lower) )
    X_zero = np.hstack( (x_zero, t_zero) )

    # each one of these three arrays haS 2 columns, 
    # now we stack them vertically, the resulting array will also have 2 
    # columns and 100 rows:
    X_u_train = np.vstack( (X_upper, X_lower, X_zero) )


    # shuffle X_u_train:
    index = np.arange(0, N_u)
    np.random.shuffle(index)
    X_u_train = X_u_train[index, :]
    
    # make X_f_train:
    X_f_train = np.zeros((N_f, 2), dtype=float)
    for row in range(N_f):
        x = random.uniform(-1, 1)  # x range
        t = random.uniform( 0, 1)  # t range

        X_f_train[row, 0] = x 
        X_f_train[row, 1] = t

    # add the boundary points to the collocation points:
    X_f_train = np.vstack( (X_f_train, X_u_train) )

    # make u_train
    u_upper =  np.zeros((N_u//4, 1), dtype=float)
    u_lower =  np.zeros((N_u//4, 1), dtype=float) 
    u_zero = -np.sin(np.pi * x_zero)  

    # stack them in the same order as X_u_train was stacked:
    u_train = np.vstack( (u_upper, u_lower, u_zero) )

    # match indices with X_u_train
    u_train = u_train[index, :]
    #x, y, val = get_data(1024, 256, 2, 2*PI, 0.01)
    net = PINN(num_layers=8, hidden_dim=20, output_dim=1, act_fn=tf.nn.tanh)
    sup = BurgerSupervisor(nu=0.01, net=net, epochs=1000, batch_size=100, lr=0.01, alpha=1)
    x = (X_u_train, X_f_train) 
    a = sup.train(x, u_train, X_u_train, u_train)
    # out = net(x[0])
    # print(out.shape)