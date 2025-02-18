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
from scipy.io import loadmat
from scipy.interpolate import interpn
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
import h5py

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

def equ_func3(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.vjp
    out, out_t = jax.vjp(u_t, g_batch)
    return out, out_t(jnp.ones(g_batch.shape))

def equ_func4(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.vjp(u_t, g_batch)
    return out, out_t(jnp.ones(g_batch.shape))

def acc_cal(dynamic_params, all_params, g_batch, model_fns):
    all_params["network"]["layers"] = dynamic_params
    weights = all_params["problem"]["loss_weights"]
    out, out_t = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_x = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_y = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_z = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]

    ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["data"]["domain_range"]["t"][1]
    vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["data"]["domain_range"]["t"][1]
    wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["data"]["domain_range"]["t"][1]

    ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["data"]["domain_range"]["x"][1]
    vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["data"]["domain_range"]["x"][1]
    wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["data"]["domain_range"]["x"][1]
    px = all_params["data"]['u_ref']*out_x[:,3:4]/all_params["data"]["domain_range"]["x"][1]

    uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["data"]["domain_range"]["y"][1]
    vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["data"]["domain_range"]["y"][1]
    wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["data"]["domain_range"]["y"][1]
    py = all_params["data"]['u_ref']*out_y[:,3:4]/all_params["data"]["domain_range"]["y"][1]

    uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["data"]["domain_range"]["z"][1]
    vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["data"]["domain_range"]["z"][1]
    wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["data"]["domain_range"]["z"][1]
    pz = all_params["data"]['u_ref']*out_z[:,3:4]/all_params["data"]["domain_range"]["z"][1]
    

    acc_x = ut + u*ux + v*uy + w*uz
    acc_y = vt + u*vx + v*vy + w*vz
    acc_z = wt + u*wx + v*wy + w*wz

    return acc_x, acc_y, acc_z

class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data

#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    import argparse
    from glob import glob
    checkpoint_fol = "TBL_run_test61"
    #parser = argparse.ArgumentParser(description='TBL_PINN')
    #parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint', default="")
    #args = parser.parse_args()
    #checkpoint_fol = args.checkpoint
    #print(checkpoint_fol, type(checkpoint_fol))
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a['data_init_kwargs']['path'] = '/scratch/hyun/TBL/'
    a['problem_init_kwargs']['path_s'] = '/scratch/hyun/Ground/'


    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)
    #checkpoint_list = np.sort(glob(run.c.model_out_dir+'/*.pkl'))
    #with open(run.c.model_out_dir + "saved_dic_720000.pkl","rb") as f:

#%%    
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[4].split('.')[0]))
    print(checkpoint_list)
    with open(checkpoint_list[-1],'rb') as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params

    u_tau = 15*10**(-6)/36.2/10**(-6)
    u_ref_n = 4.9968*10**(-2)/u_tau
    delta = 36.2*10**(-6)
    x_ref_n = 1.0006*10**(-3)/delta
#%%
    timestep = 24
    pos_ref = all_params["domain"]["in_max"].flatten()
    vel_ref = np.array([all_params["data"]["u_ref"],
                        all_params["data"]["v_ref"],
                        all_params["data"]["w_ref"]])
    ref_key = ['t_ref', 'x_ref', 'y_ref', 'z_ref', 'u_ref', 'v_ref', 'w_ref']
    ref_data = {ref_key[i]:ref_val for i, ref_val in enumerate(np.concatenate([pos_ref,vel_ref]))}
    datapath = '/scratch/hyun/PG_TBL_dnsinterp.mat'
    data = loadmat(datapath)
    eval_key = ['x', 'y', 'z', 'x_pred', 'y_pred', 'z_pred', 'u1', 'v1', 'w1', 'p1', 'um', 'vm', 'wm']
    DNS_grid = (0.001*data['y'][:,0,0], 0.001*data['x'][0,:,0], 0.001*data['z'][0,0,:])
    eval_grid = np.concatenate([0.001*data['y_pred'].reshape(32,88,410)[:31,:,:].reshape(-1,1),
                                0.001*data['x_pred'].reshape(32,88,410)[:31,:,:].reshape(-1,1),
                                0.001*data['z_pred'].reshape(32,88,410)[:31,:,:].reshape(-1,1)],1)
    vel_ground = [interpn(DNS_grid, data[eval_key[i+6]], eval_grid).reshape(31,88,410) for i in range(3)]
    fluc_ground = [interpn(DNS_grid, data[eval_key[i+6]], eval_grid).reshape(31,88,410) - data[eval_key[i+10]].reshape(32,88,410)[:31,:,:] for i in range(3)]
    p_cent = interpn(DNS_grid, data['p1'], eval_grid).reshape(31,88,410)
    fluc_ground.append(p_cent-np.mean(p_cent))
    V_mag = np.sqrt(fluc_ground[0]**2+fluc_ground[1]**2+fluc_ground[2]**2)
    div_list = [V_mag, V_mag, V_mag, fluc_ground[-1]]
    fluc_ground[-1] = (fluc_ground[-1]*u_ref_n**2 - 0.0025*eval_grid[:,1].reshape(31,88,410)[0,0,:]*x_ref_n)/u_ref_n**2
    eval_grid_n = np.concatenate([np.zeros((eval_grid.shape[0],1))+timestep*(1/17594),
                                eval_grid[:,1:2], eval_grid[:,0:1], eval_grid[:,2:3]],1)
    for i in range(eval_grid_n.shape[1]): eval_grid_n[:,i] = eval_grid_n[:,i]/ref_data[ref_key[i]] 

#%%
    p_cent2 = p_cent - 0.0025*eval_grid[:,1].reshape(31,88,410)[0,0,:]*x_ref_n/u_ref_n**2
    p_cent2 = p_cent2 - np.mean(p_cent2)
#%%
    p_cent3 = p_cent - 0.0025*eval_grid[:,1].reshape(31,88,410)[0,0,:]*1.185*x_ref_n/u_ref_n**2
    p_cent3 = p_cent3 - np.mean(p_cent3)

#%%
    pred = np.concatenate([model_fn(all_params, eval_grid_n[10000*i:10000*(i+1),:]) for i in range(eval_grid_n.shape[0]//10000+1)],0)
    ref_key = ['t_ref', 'x_ref', 'y_ref', 'z_ref', 'u_ref', 'v_ref', 'w_ref', 'u_ref']
    pred_ = np.concatenate([pred[:,i:i+1]*ref_data[ref_key[i+4]] for i in range(4)],1)
    pred_[:,-1] = pred_[:,-1]*1.185
    pred_[:,-1] = pred_[:,-1] - np.mean(pred_[:,-1])
    fluc_pred = [pred_[:,i].reshape(31,88,410) - data[eval_key[i+10]].reshape(32,88,410)[:31,:,:] for i in range(3)]
#%%
    f = np.concatenate([(fluc_pred[0]-fluc_ground[0]).reshape(-1,1), 
                        (fluc_pred[1]-fluc_ground[1]).reshape(-1,1), 
                        (fluc_pred[2]-fluc_ground[2]).reshape(-1,1)],1)
    div = np.concatenate([fluc_ground[0].reshape(-1,1), fluc_ground[1].reshape(-1,1), 
                        fluc_ground[2].reshape(-1,1)],1)
    print(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))
    print(np.linalg.norm(pred_[:,-1:].reshape(31,88,410)-p_cent3)/np.linalg.norm(p_cent3))
#%%
    def error_metric(pred, test, div):
        out = np.linalg.norm(pred-test)/np.linalg.norm(div)
        return out

    def error_metric2(pred, test):
        f = np.concatenate([(pred[0]-test[0]).reshape(-1,1),
                            (pred[1]-test[1]).reshape(-1,1),
                            (pred[2]-test[2]).reshape(-1,1)],1)
        div = np.concatenate([(test[0]).reshape(-1,1),
                            (test[1]).reshape(-1,1),
                            (test[2]).reshape(-1,1)],1)
        return np.linalg.norm(f, ord='fro')/np.linalg.norm(div, ord='fro')

    def NRMSE(pred, test, div):
        out = np.sqrt(np.mean(np.square(pred-test))/np.mean(np.square(div)))
        return out
#%%
    print('--------for_norm-------')
    print_key = ['u_error', 'v_error', 'w_error', 'p_error']
    for i in range(len(fluc_pred)): print(f"{print_key[i]} : {error_metric(fluc_pred[i], fluc_ground[i], div_list[i])}")
    print('velocity error : ', error_metric2(fluc_pred[0:3], fluc_ground[0:3]))
    print('--------NRMSE----------')
    print_key = ['u_error', 'v_error', 'w_error', 'p_error']
    for i in range(len(fluc_pred)): print(f"{print_key[i]} : {NRMSE(fluc_pred[i], fluc_ground[i], div_list[i])}")
    print('velocity error : ', np.sqrt(np.mean(np.square(np.sqrt(fluc_pred[0]**2+fluc_pred[1]**2+fluc_pred[2]**2)
                                                        -np.sqrt(fluc_ground[0]**2+fluc_ground[1]**2+fluc_ground[2]**2)))/np.mean(fluc_ground[0]**2+fluc_ground[1]**2+fluc_ground[2]**2)))
#%%
    plt.imshow(fluc_pred[1][0,:,:],cmap='jet')
    plt.colorbar()
    plt.show()
    plt.imshow(fluc_ground[1][0,:,:],cmap='jet')
    plt.colorbar()
    plt.show()
#%%
    plt.figure(figsize=(10,8))
    plt.plot(np.mean(fluc_pred[2],(0,1)),label='optimized model')
    plt.plot(np.mean(fluc_ground[2],(0,1)), label='previous model')
    plt.legend(loc="upper right", bbox_to_anchor=(1.25,0.5))
    plt.show()
#%%
    plt.plot(np.mean(pred_[:,-1:].reshape(31,88,410)-np.mean(pred_[:,-1:].reshape(31,88,410)),(0,1)))
    plt.plot(np.mean((p_cent3 - np.mean(p_cent3)),(0,1)))
    plt.plot(np.mean(p_cent2 - np.mean(p_cent2),(0,1)))
    plt.ylim(-1.2,1.4)
    plt.show()
#%%
from itertools import chain
#%%
    plt.imshow(pred_[:,-1].reshape(31,88,410)[10,:,:],cmap='jet')
    plt.colorbar()
    plt.show()
    plt.imshow(p_cent3[10,:,:],cmap='jet')
    plt.colorbar()
    plt.show()    
#%%
    print(pred_.shape[0]/410/88)
#%%
    fluc_pred = 
#%%
    print(pred.shape, fluc_ground[0].shape)
#%% temporal error는 51개의 시간단계에대해서 [:,0]는 velocity error, [:,1]은 pressure error

    output_shape = (213,141,61)
    temporal_error_vel_list = []
    temporal_error_pre_list = []
    for j in range(51):
        print(j)
        pred = model_fn(all_params, valid_data['pos'].reshape((51,)+output_shape+(4,))[j,:,:,:,:].reshape(-1,4))
        output_keys = ['u', 'v', 'w', 'p']
        output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                        all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]
        outputs = {output_keys[i]:pred[:,i]*output_unnorm[i] for i in range(len(output_keys))}
        outputs['p'] = outputs['p'] - np.mean(outputs['p'])
        output_ext = {output_keys[i]:valid_data['vel'].reshape((51,)+output_shape+(4,))[j,:,:,:,i].reshape(-1) for i in range(len(output_keys))}
        output_ext['p'] = output_ext['p'] - np.mean(output_ext['p'])


        f = np.concatenate([(outputs['u']-output_ext['u']).reshape(-1,1), 
                            (outputs['v']-output_ext['v']).reshape(-1,1), 
                            (outputs['w']-output_ext['w']).reshape(-1,1)],1)
        div = np.concatenate([output_ext['u'].reshape(-1,1), output_ext['v'].reshape(-1,1), 
                            output_ext['w'].reshape(-1,1)],1)

        temporal_error_pre_list.append(np.linalg.norm(outputs['p'] - output_ext['p'])/np.linalg.norm(output_ext['p']))
        temporal_error_vel_list.append(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))    

    temporal_error = np.concatenate([np.array(temporal_error_vel_list).reshape(-1,1),
                                     np.array(temporal_error_pre_list).reshape(-1,1)],1)

    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)

    with open("datas/"+checkpoint_fol+"/temporal_error.pkl","wb") as f:
        pickle.dump(temporal_error,f)
    f.close()

# %%
def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
    return out_xx

def equ_func2(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out_t
print(equ_func(all_params, eval_grid_n[0:10000,:],jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(eval_grid_n[0:10000,:].shape[0],1)),model_fn))
#%%
a = model_fn(all_params, eval_grid_n[0:10000,:])
# %%
from jax import vjp
def ls(x):
    aa = model_fn(all_params, x)
    return aa
def ds(x):
    aa = jnp.concatenate([x[:,0:1]*x[:,1:2]*x[:,2:3]*x[:,3:4],
                         x[:,0:1]**2*x[:,1:2]*x[:,2:3]*x[:,3:4],
                         x[:,0:1]*x[:,1:2]**2*x[:,2:3]*x[:,3:4],
                         x[:,0:1]*x[:,1:2]*x[:,2:3]*x[:,3:4]**2],1)
    return aa

def jvptest(x, cotangent):
    def u_t(x):
        return jax.jvp(ds, (x,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_t, (x, ), (cotangent, ))
    return out_xx

def vgrad(fn, x):
    def u_t(x):
        y, vjp_fn = vjp(ls, x)
        return vjp_fn(jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(x.shape[0],1)))[0]
    
    y2, vjp_fn2 = vjp(u_t, x)
    return u_t(x), vjp_fn2(jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(x.shape[0],1)))[0]
def jacrevs(x):
    f_j = jax.jacrev(ds)(x)
    return f_j
#%%
print(jvptest(xx, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(xx.shape[0],1))))
#%%
print(jacrevs(xx).shape)
print(jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(xx.shape[0],1))@jacrevs(xx))
#%%
print(vgrad(model_fn, eval_grid_n[0:10000,:]))
# %%
xx = jnp.array([[1.0, 2.0, 3.0, 4.0],[2.0, 6.0, 4.0, 9.0],[3.0, 7.0, 1.0, 5.0]])
#%%
ad = jnp.array([[1.1,2.2],[2.3,3.4]])
# %%
print(xx)
print(jnp.ones((2, 2)))
# %%
print(all_params['network']['layers'][-1][1])
#%%
