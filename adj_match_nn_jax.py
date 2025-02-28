import jax
import numpy as np
import jax.numpy as jnp
import optax
import pickle
import os

from functools import partial
from jax import vmap, grad, jit, random

class MLP:
    def __init__(self, layers, in_dim, out_dim, act_fn) -> None:
        self.layers = [in_dim] + layers + [out_dim]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = self.init_network(random.PRNGKey(5))
        self.act_fn = act_fn

    def init_network(self, key):
        initializer = jax.nn.initializers.glorot_normal()
        keys = random.split(key, len(self.layers))
        def init_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return initializer(w_key, (n, m)), random.normal(b_key, (n,))
        return [init_layer(m, n, k) for m, n, k in zip(self.layers[:-1], self.layers[1:], keys)]

    def activation(self, x):
        if self.act_fn == 'relu':
            return jax.nn.relu(x)
        elif self.act_fn == 'tanh':
            return jnp.tanh(x)
        elif self.act_fn == 'sigmoid':
            return jax.nn.sigmoid(x)
        elif self.act_fn == 'gelu':
            return jax.nn.gelu(x)
        else:
            return x
    
    def forward(self, params, x):
        inputs = x
        for w, b in params[:-1]:
            inputs = jnp.dot(w, inputs) + b
            inputs = self.activation(inputs)
        w_f, b_f = params[-1]
        out = jnp.dot(w_f, inputs) + b_f
        return out
    
    def apply(self, params, x):
        f_pass_v = vmap(self.forward, in_axes=(None, 0))
        return f_pass_v(params, x)
    
    def nn_adjoint(self, params, x):
        def adjoint(params, x):
            jac = jax.jacfwd(self.forward, argnums=1)(params, x)
            return jac
        return vmap(adjoint, in_axes=(None, 0), out_axes=0)(params, x)

class Trainer:
    def __init__(self, net, num_epochs, batch_size, learning_rate, optimizer):
        self.net = net
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.data_path = './Data/mixed_init_cond/'

    def load_data(self, path, mode):
        if mode == 'train':
            files = os.listdir(path)[:-1]
        else:
            files = [os.listdir(path)[-1]]
        print(files)
        x_list = []
        y_list = []
        adj_list = []
        for f in files:
            with open(path + f, 'rb') as f:
                sol, adj = pickle.load(f)
            sol = np.stack(sol, axis=1)
            adj = np.transpose(np.stack(adj, axis=-1), axes=[2, 0, 1])
            x = sol[..., :-1].T
            y = sol[..., 1:].T 
            x_list.append(x)
            y_list.append(y)
            adj_list.append(adj)
        return np.concatenate(x_list), np.concatenate(y_list), np.concatenate(adj_list) 

    def prepare_data(self, mode):
        """
        use 60 % of the time to train, 20 validation and rest 20 for testing.
        """
        x_, y_, adj_ = self.load_data(self.data_path, mode)
        rs = np.random.RandomState(0)
        idx = rs.permutation(x_.shape[0]) 

        x = x_[idx]
        y = y_[idx]
        adj = adj_[idx]
        
        train_len = int(.6 * x.shape[0])
        x_train = x[:train_len]
        y_train = y[:train_len]
        adj_train = adj[:train_len]

        x_val = x[train_len:]
        y_val = y[train_len:]
        adj_val = adj[train_len:]

        train_data = {'x': x_train, 'y': y_train, 'adj': adj_train}
        val_data = {'x': x_val, 'y': y_val, 'adj': adj_val}
        if mode == 'train':
            return train_data, val_data
        else:
            return {'x': x_, 'y': y_, 'adj': adj_}



    def loss(self, params, x, y, adj_y, alpha):
        
        pred = self.net.apply(params, x)
        adj = self.net.nn_adjoint(params, x)
        # pred = self.net.forward(params, x, t)
        totLoss = jnp.mean((pred - y)**2) + alpha*jnp.mean((adj - adj_y)**2)
        return totLoss

    @partial(jax.jit, static_argnums=(0,)) 
    def step_(self, params, x, y, adj_y, alpha, opt_state):
        ls = self.loss(params, x, y, adj_y, alpha)
        grads = jax.grad(self.loss, argnums=0)(params, x, y, adj_y, alpha)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, ls, opt_state

    def train_(self, params, train_data, val_data, alpha):
        x_train = train_data['x']
        adj_train = train_data['adj']
        y_train = train_data['y']

        x_val = val_data['x']
        adj_val = val_data['adj']
        y_val = val_data['y']


        logger = {'train_loss': [], 'val_loss': []}
        opt_state = self.optimizer.init(params)
        for ep in range(self.num_epochs):
            train_running_ls = []
            for i in range(len(x_train)//self.batch_size):
                x_batch = x_train[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
                adj_batch = adj_train[i*self.batch_size:(i+1)*self.batch_size]
                params, ls, opt_state= self.step_(params, x_batch, y_batch,  adj_batch, alpha, opt_state)
                train_running_ls.append(ls)
            ls_val = self.loss(params, x_val, y_val, adj_val, alpha=1) # set alpha=1 in validation.
            logger['train_loss'].append(jnp.asarray(train_running_ls).mean())
            logger['val_loss'].append(ls_val)
            with open('./logs/logger', 'wb') as f:
                pickle.dump(logger, f)
            print('Epoch: {} trianing loss {} validation loss {}'.format(ep, logger['train_loss'][-1], logger['val_loss'][-1]))
        return params    
    
    def predict(self, params, x):
        pred = self.net.apply(params, x)
        return pred



if __name__ == "__main__":
    save_name = 'mixed_init'
    net = MLP([20]*10, in_dim=128, out_dim=128, act_fn='tanh')
     
    sup = Trainer(net=net, num_epochs=1000, batch_size=16, learning_rate=0.01, optimizer=optax.adam(learning_rate=0.001))
    train, val = sup.prepare_data('train')
    sup.train_(net.params, train, val, 1)
    
