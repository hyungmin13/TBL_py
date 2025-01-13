#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
    return out_x, out_xx

def equ_func2(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def PINN_loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
    all_params["network"]["layers"] = dynamic_params
    weights = all_params["problem"]["loss_weights"]
    domain_range = all_params["data"]["domain_range"]
    u_ref, v_ref, w_ref = all_params["data"]['u_ref'], all_params["data"]['v_ref'], all_params["data"]['w_ref']
    viscosity = all_params["data"]["viscosity"]

    def scale_output(out, ref, dim):
        return ref * out[:, dim:dim+1]

    def scale_derivative(out, ref, dim, axis):
        return ref * out[:, dim:dim+1] / domain_range[axis][1]

    def scale_second_derivative(out, ref, dim, axis):
        return ref * out[:, dim:dim+1] / domain_range[axis][1]**2

    out, out_t = equ_func2(all_params, g_batch, jnp.array([[1.0, 0.0, 0.0, 0.0]] * g_batch.shape[0]), model_fns)
    out_x, out_xx = equ_func(all_params, g_batch, jnp.array([[0.0, 1.0, 0.0, 0.0]] * g_batch.shape[0]), model_fns)
    out_y, out_yy = equ_func(all_params, g_batch, jnp.array([[0.0, 0.0, 1.0, 0.0]] * g_batch.shape[0]), model_fns)
    out_z, out_zz = equ_func(all_params, g_batch, jnp.array([[0.0, 0.0, 0.0, 1.0]] * g_batch.shape[0]), model_fns)

    p_out = model_fns(all_params, particles)
    b_out = model_fns(all_params, boundaries)

    u, v, w = scale_output(out, u_ref, 0), scale_output(out, v_ref, 1), scale_output(out, w_ref, 2)
    ut, vt, wt = scale_derivative(out_t, u_ref, 0, 't'), scale_derivative(out_t, v_ref, 1, 't'), scale_derivative(out_t, w_ref, 2, 't')
    ux, vx, wx, px = scale_derivative(out_x, u_ref, 0, 'x'), scale_derivative(out_x, v_ref, 1, 'x'), scale_derivative(out_x, w_ref, 2, 'x'), scale_derivative(out_x, u_ref, 3, 'x')
    uy, vy, wy, py = scale_derivative(out_y, u_ref, 0, 'y'), scale_derivative(out_y, v_ref, 1, 'y'), scale_derivative(out_y, w_ref, 2, 'y'), scale_derivative(out_y, u_ref, 3, 'y')
    uz, vz, wz, pz = scale_derivative(out_z, u_ref, 0, 'z'), scale_derivative(out_z, v_ref, 1, 'z'), scale_derivative(out_z, w_ref, 2, 'z'), scale_derivative(out_z, u_ref, 3, 'z')
    uxx, vxx, wxx = scale_second_derivative(out_xx, u_ref, 0, 'x'), scale_second_derivative(out_xx, v_ref, 1, 'x'), scale_second_derivative(out_xx, w_ref, 2, 'x')
    uyy, vyy, wyy = scale_second_derivative(out_yy, u_ref, 0, 'y'), scale_second_derivative(out_yy, v_ref, 1, 'y'), scale_second_derivative(out_yy, w_ref, 2, 'y')
    uzz, vzz, wzz = scale_second_derivative(out_zz, u_ref, 0, 'z'), scale_second_derivative(out_zz, v_ref, 1, 'z'), scale_second_derivative(out_zz, w_ref, 2, 'z')

    def compute_loss(pred, true):
        return jnp.mean((pred - true) ** 2)

    loss_u = compute_loss(u_ref * p_out[:, 0:1], particle_vel[:, 0:1])
    loss_v = compute_loss(v_ref * p_out[:, 1:2], particle_vel[:, 1:2])
    loss_w = compute_loss(w_ref * p_out[:, 2:3], particle_vel[:, 2:3])
    loss_b_u = compute_loss(u_ref * b_out[:, 0:1], 0)
    loss_b_v = compute_loss(v_ref * b_out[:, 1:2], 0)
    loss_b_w = compute_loss(w_ref * b_out[:, 2:3], 0)
    loss_con = compute_loss(ux + vy + wz, 0)

    def compute_NS_loss(t, u, v, w, p, uxx, uyy, uzz):
        return compute_loss(t + u * ux + v * uy + w * uz + p - viscosity * (uxx + uyy + uzz) - 2.22e-1 / (3 * 0.43685 ** 2) * u, 0)

    loss_NS1 = compute_NS_loss(ut, u, v, w, px, uxx, uyy, uzz)
    loss_NS2 = compute_NS_loss(vt, u, v, w, py, vxx, vyy, vzz)
    loss_NS3 = compute_NS_loss(wt, u, v, w, pz, wxx, wyy, wzz)

    total_loss = (weights[0] * loss_u + weights[1] * loss_v + weights[2] * loss_w + weights[3] * loss_con +
                  weights[4] * loss_NS1 + weights[5] * loss_NS2 + weights[6] * loss_NS3 +
                  weights[7] * (loss_b_u + loss_b_v + loss_b_w))

    return total_loss

@partial(jax.jit, static_argnums=(1, 4, 9))
def PINN_update(model_states, optimiser_fn, dynamic_params, static_params, static_keys, grids, particles, particle_vel, boundaries, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(PINN_loss, argnums=0)(dynamic_params, all_params, grids, particles, particle_vel, boundaries, model_fn)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    return lossval, model_states, dynamic_params
class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
class PINN(PINNbase):
    def train(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network.network_fn
        dynamic_params = all_params["network"].pop("layers")
        key, batch_key = random.split(key)
        g_batch_key1, g_batch_key2, g_batch_key3, g_batch_key4, p_batch_key, b_batch_key, e_batch_key = random.split(batch_key, num = 7)
        g_batch_keys1 = random.split(g_batch_key1, num = self.c.optimization_init_kwargs["n_steps"])
        g_batch_keys2 = random.split(g_batch_key2, num = self.c.optimization_init_kwargs["n_steps"])
        g_batch_keys3 = random.split(g_batch_key3, num = self.c.optimization_init_kwargs["n_steps"])
        g_batch_keys4 = random.split(g_batch_key4, num = self.c.optimization_init_kwargs["n_steps"])

        p_batch_keys = random.split(p_batch_key, num = self.c.optimization_init_kwargs["n_steps"])
        b_batch_keys = random.split(b_batch_key, num = self.c.optimization_init_kwargs["n_steps"])
        e_batch_keys = random.split(e_batch_key, num = self.c.optimization_init_kwargs["n_steps"]//2000)

        p_batch_keys = iter(p_batch_keys)
        g_batch_keys1 = iter(g_batch_keys1)
        g_batch_keys2 = iter(g_batch_keys2)
        g_batch_keys3 = iter(g_batch_keys3)
        g_batch_keys4 = iter(g_batch_keys4)
        b_batch_keys = iter(b_batch_keys)
        e_batch_keys = iter(e_batch_keys)
        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        ab = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        ac = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        ad = (ac, treedef)

        p_key = next(p_batch_keys)
        g_key1 = next(g_batch_keys1)
        g_key2 = next(g_batch_keys2)
        g_key3 = next(g_batch_keys3)
        g_key4 = next(g_batch_keys4)
        b_key = next(b_batch_keys)
        p_batch = random.choice(p_key,train_data['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
        v_batch = random.choice(p_key,train_data['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))

        g_batch = jnp.stack([random.choice(g_key1,grids['eqns']['t'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                            random.choice(g_key2,grids['eqns']['x'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                            random.choice(g_key3,grids['eqns']['y'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                            random.choice(g_key4,grids['eqns']['z'],shape=(self.c.optimization_init_kwargs["e_batch"],))],axis=1)
        b_batch = jnp.stack([random.choice(b_key,grids['bczl']['t'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                             random.choice(b_key,grids['bczl']['x'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                             random.choice(b_key,grids['bczl']['y'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                             random.choice(b_key,np.array([0]),shape=(self.c.optimization_init_kwargs["e_batch"],))],axis=1)
        update = PINN_update.lower(model_states, optimiser_fn, dynamic_params, ab, ad, g_batch, p_batch, v_batch, b_batch, model_fn).compile()
        
        for i in tqdm(range(self.c.optimization_init_kwargs["n_steps"])):
            template = ("iteration {}, loss_val {}")
            p_key = next(p_batch_keys)
            g_key1 = next(g_batch_keys1)
            g_key2 = next(g_batch_keys2)
            g_key3 = next(g_batch_keys3)
            g_key4 = next(g_batch_keys4)

            p_batch = random.choice(p_key,train_data['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            v_batch = random.choice(p_key,train_data['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            g_batch = jnp.stack([random.choice(g_key1,grids['eqns']['t'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                                random.choice(g_key2,grids['eqns']['x'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                                random.choice(g_key3,grids['eqns']['y'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                                random.choice(g_key4,grids['eqns']['z'],shape=(self.c.optimization_init_kwargs["e_batch"],))],axis=1)            
            lossval, model_states, dynamic_params = update(model_states, dynamic_params, ab, g_batch, p_batch, v_batch, b_batch)
        
        
            self.report(i, model_states, dynamic_params, all_params, p_batch, v_batch, valid_data, e_batch_keys, model_fn)

    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data
    def report(self, i, model_states, dynamic_params, all_params, p_batch, v_batch, valid_data, e_batch_key, model_fns):
        model_save = (i % 20000 == 0)
        if model_save:
            all_params["network"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            v_pred2 = model_fns(all_params, e_batch_pos)
            p_new = 1.185*all_params["data"]["u_ref"]*v_pred2[:,3:]-(jnp.mean(1.185*all_params["data"]["u_ref"]*v_pred2[:,3:] - e_batch_vel[:,3:]))
            u_error = jnp.sqrt(jnp.mean((all_params["data"]["u_ref"]*v_pred2[:,0:1] - e_batch_vel[:,0:1])**2)/jnp.mean(e_batch_vel[:,0:1]**2))
            v_error = jnp.sqrt(jnp.mean((all_params["data"]["v_ref"]*v_pred2[:,1:2] - e_batch_vel[:,1:2])**2)/jnp.mean(e_batch_vel[:,1:2]**2))
            w_error = jnp.sqrt(jnp.mean((all_params["data"]["w_ref"]*v_pred2[:,2:3] - e_batch_vel[:,2:3])**2)/jnp.mean(e_batch_vel[:,2:3]**2))
            p_error = jnp.sqrt(jnp.mean((p_new - e_batch_vel[:,3:])**2)/jnp.mean(e_batch_vel[:,3:4]**2))
            v_pred = model_fns(all_params, p_batch)
            u_loss = jnp.mean((all_params["data"]["u_ref"]*v_pred[:,0:1] - v_batch[:,0:1])**2)
            v_loss = jnp.mean((all_params["data"]["v_ref"]*v_pred[:,1:2] - v_batch[:,1:2])**2)
            w_loss = jnp.mean((all_params["data"]["w_ref"]*v_pred[:,2:3] - v_batch[:,2:3])**2)
            model = Model(all_params["network"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
            
            print(u_loss, v_loss, w_loss, u_error, v_error, w_error, p_error)

        return

#%%
if __name__=="__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    run = "TBL_run_00"
    all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}

    # Set Domain params
    frequency = 17594
    domain_range = {'t':(0,50/frequency), 'x':(0,0.056), 'y':(0,0.012), 'z':(0,0.00424)}
    grid_size = [51, 2800, 600, 212]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']

    # Set Data params
    path = '/scratch/hyun/TBL/'
    timeskip = 1
    track_limit = 424070
    viscosity = 15*10**(-6)
    data_keys = ['pos', 'vel', 'acc', 'track']
    
    # Set network params
    key = random.PRNGKey(1)
    layer_sizes = [4, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 4]
    network_name = 'MLP'

    # Set problem params
    viscosity = 15e-6
    loss_weights = (1.0, 1.0, 1.0, 0.0000001, 0.00000001, 0.00000001, 0.00000001, 1.0)
    path_s = '/scratch/hyun/Ground/'
    problem_name = 'TBL'
    # Set optimization params
    n_steps = 100000000
    optimiser = optax.adam
    learning_rate = 1e-2
    p_batch = 10000
    e_batch = 10000
    b_batch = 10000


    c = Constants(
        run= run,
        domain_init_kwargs = dict(domain_range = domain_range, frequency = frequency, 
                                  grid_size = grid_size, bound_keys=bound_keys),
        data_init_kwargs = dict(path = path, domain_range = domain_range, timeskip = timeskip,
                                track_limit = track_limit, frequency = frequency, data_keys = data_keys, viscosity = viscosity),
        network_init_kwargs = dict(key = key, layer_sizes = layer_sizes, network_name = network_name),
        problem_init_kwargs = dict(domain_range = domain_range, viscosity = viscosity, loss_weights = loss_weights,
                                   path_s = path_s, frequency = frequency, problem_name = problem_name),
        optimization_init_kwargs = dict(optimiser = optimiser, learning_rate = learning_rate, n_steps = n_steps,
                                        p_batch = p_batch, e_batch = e_batch, b_batch = b_batch)

    )
    run = PINN(c)
    run.train()
    #all_params, model_fn, train_data, valid_data = run.test()

