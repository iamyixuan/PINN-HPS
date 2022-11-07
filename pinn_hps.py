import numpy as np
import tensorflow as tf
from deephyper.problem import HpProblem
from PINN import PINN, BurgerSupervisor, get_data
from deephyper.search.hps import CBO

# define the search space
problem = HpProblem()
problem.add_hyperparameter((5, 20), "num_layers", default_value=5)
problem.add_hyperparameter((1e-5, 1e-2), "lr", default_value=0.01)
problem.add_hyperparameter((5, 20), "hidden_dim", default_value=5)
problem.add_hyperparameter((2, 12), "batch_size", default_value=2)
problem.add_hyperparameter((10, 15), "epochs", default_value=10)
problem.add_hyperparameter((0., 1.), "alpha", default_value=.5)

x, y, val = get_data(1024, 256, 2, 2*3.14, 0.01)

# define the run function
def run(config):
    num_layers = config["num_layers"] 
    hidden_dim = config["hidden_dim"]
    output_dim = 1
    nu = 0.01
    epochs = config["epochs"] 
    batch_size = config["batch_size"]
    lr = config["lr"]
    alpha = config["alpha"]
    act_fn = tf.nn.relu
    net = PINN(num_layers, hidden_dim, output_dim, act_fn)
    sup = BurgerSupervisor(nu, net, epochs, batch_size, lr, alpha)
    val_r2, val_f_loss = sup.train(x, y, val[0][:10], val[1][:10])
    return val_r2

search = CBO(problem, run, initial_points=[problem.default_configuration], log_dir="cbo-results", random_state=42)
results = search.search(max_evals=10)
