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
import h5py

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
    acc = np.concatenate([acc_x.reshape(-1,1), acc_y.reshape(-1,1), acc_z.reshape(-1,1)],1)
    return acc
#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    import argparse
    from glob import glob
    #checkpoint_fol = "TBL_SOAP_k1"
    parser = argparse.ArgumentParser(description='TBL_PINN')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint', default="")
    args = parser.parse_args()
    checkpoint_fol = args.checkpoint
    print(checkpoint_fol, type(checkpoint_fol))
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a['data_init_kwargs']['path'] = 'DNS/'
    a['problem_init_kwargs']['path_s'] = 'Ground/'


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
    
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[4].split('.')[0]))
    print(checkpoint_list)
    with open(checkpoint_list[-1],'rb') as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()
    all_params['data']['track_limit'] = 424070
    valid2_data, all_params_ = Data.train_data(all_params)
    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params
    dynamic_params = all_params["network"]["layers"]
    indexes, counts = np.unique(train_data['pos'][:,0], return_counts=True)
    indexes2, counts2 = np.unique(valid2_data['pos'][:,0], return_counts=True)
    pos_v = []
    vel_v = []
    acc_v = []
    c = 0

    for i in range(len(counts2)):
        pos_v.append(valid2_data['pos'][c+210750:c+counts2[i],:])
        vel_v.append(valid2_data['vel'][c+210750:c+counts2[i],:])
        acc_v.append(valid2_data['acc'][c+210750:c+counts2[i],:])
        c = c+counts2[i]
    pos_v = np.concatenate(pos_v,0)
    vel_v = np.concatenate(vel_v,0)
    acc_v = np.concatenate(acc_v,0)
    indexes2, counts2 = np.unique(pos_v[:,0], return_counts=True)

#%%
    u_tau = 15*10**(-6)/36.2/10**(-6)
    u_ref_n = 4.9968*10**(-2)/u_tau
    delta = 36.2*10**(-6)
    x_ref_n = 1.0006*10**(-3)/delta

#%% temporal error는 51개의 시간단계에대해서 [:,0]는 velocity error, [:,1]은 pressure error
    output_shape = (213,141,61)
    temporal_error_vel_list = []
    temporal_error_pre_list = []
    temporal_error_acc_list = []
    temporal_error_acc_v_list = []
    acc_v_list = []
    for j in range(counts.shape[0]):
        print(j)
        acc = np.concatenate([acc_cal(all_params["network"]["layers"], all_params, train_data['pos'][np.sum(counts[:j]):np.sum(counts[:(j+1)])][10000*s:10000*(s+1)], model_fn) 
                              for s in range(train_data['pos'][np.sum(counts[:j]):np.sum(counts[:(j+1)])].shape[0]//10000+1)],0)
        acc_pred_v = np.concatenate([acc_cal(all_params["network"]["layers"], all_params, pos_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])][10000*s:10000*(s+1)], model_fn) 
                              for s in range(pos_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])].shape[0]//10000+1)],0)
        pred = model_fn(all_params, valid_data['pos'].reshape((51,)+output_shape+(4,))[j,:,:,:,:].reshape(-1,4))
        output_keys = ['u', 'v', 'w', 'p']
        output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                        all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]
        outputs = {output_keys[i]:pred[:,i]*output_unnorm[i] for i in range(len(output_keys))}
        outputs['p'] = outputs['p'] - np.mean(outputs['p'])
        output_ext = {output_keys[i]:valid_data['vel'].reshape((51,)+output_shape+(4,))[j,:,:,:,i].reshape(-1) for i in range(len(output_keys))}
        output_ext['p'] = output_ext['p'] - 0.0025*all_params['domain']['in_max'][0,1]*valid_data['pos'].reshape((51,)+output_shape+(4,))[0,:,:,:,1].reshape(-1)*1.185*x_ref_n/u_ref_n**2
        output_ext['p'] = output_ext['p'] - np.mean(output_ext['p'])

        f = np.concatenate([(outputs['u']-output_ext['u']).reshape(-1,1), 
                            (outputs['v']-output_ext['v']).reshape(-1,1), 
                            (outputs['w']-output_ext['w']).reshape(-1,1)],1)
        div = np.concatenate([output_ext['u'].reshape(-1,1), output_ext['v'].reshape(-1,1), 
                            output_ext['w'].reshape(-1,1)],1)
        f2 = np.concatenate([(acc[:,0]-train_data['acc'][np.sum(counts[:j]):np.sum(counts[:(j+1)])][:,0]).reshape(-1,1), 
                             (acc[:,1]-train_data['acc'][np.sum(counts[:j]):np.sum(counts[:(j+1)])][:,1]).reshape(-1,1), 
                             (acc[:,2]-train_data['acc'][np.sum(counts[:j]):np.sum(counts[:(j+1)])][:,2]).reshape(-1,1)],1)
        div2 = np.concatenate([train_data['acc'][:,0].reshape(-1,1), train_data['acc'][:,1].reshape(-1,1), 
                               train_data['acc'][:,2].reshape(-1,1)],1)
        f3 = np.concatenate([(acc_pred_v[:,0]-acc_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])][:,0]).reshape(-1,1), 
                             (acc_pred_v[:,1]-acc_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])][:,1]).reshape(-1,1), 
                             (acc_pred_v[:,2]-acc_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])][:,2]).reshape(-1,1)],1)
        div3 = np.concatenate([acc_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])][:,0].reshape(-1,1), 
                               acc_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])][:,1].reshape(-1,1), 
                               acc_v[np.sum(counts2[:j]):np.sum(counts2[:(j+1)])][:,2].reshape(-1,1)],1)  
        temporal_error_pre_list.append(np.linalg.norm(outputs['p'] - output_ext['p'])/np.linalg.norm(output_ext['p']))
        temporal_error_vel_list.append(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))    
        temporal_error_acc_list.append(np.linalg.norm(f2, ord='fro')/np.linalg.norm(div2,ord='fro'))
        temporal_error_acc_v_list.append(np.linalg.norm(f3, ord='fro')/np.linalg.norm(div3,ord='fro'))
        acc_v_list.append(acc_pred_v)
    acc_v_list = np.concatenate(acc_v_list,0)
    print(acc_v_list.shape, acc_v.shape)
    acc_v_list = np.concatenate([acc_v_list, acc_v],1)
    temporal_error = np.concatenate([np.array(temporal_error_vel_list).reshape(-1,1),
                                     np.array(temporal_error_pre_list).reshape(-1,1),
                                     np.array(temporal_error_acc_list).reshape(-1,1),
                                     np.array(temporal_error_acc_v_list).reshape(-1,1)],1)

#%%
    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)

    with open("datas/"+checkpoint_fol+"/temporal_error.pkl","wb") as f:
        pickle.dump(temporal_error,f)
    f.close()
    with open("datas/"+checkpoint_fol+"/acc_v.pkl","wb") as f:
        pickle.dump(acc_v_list,f)