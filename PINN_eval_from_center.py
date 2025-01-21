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
    checkpoint_fol = "TBL_run_06"
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a['data_init_kwargs']['path'] = '/scratch/hyun/TBL/'
    a['problem_init_kwargs']['path_s'] = '/scratch/hyun/Ground/'
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
        pickle.dump(a,f)

    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)

    with open(run.c.model_out_dir + "saved_dic_340000.pkl","rb") as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params
#%% valid_pred_list_total 는 모든 시간 [i] 에 대해 중심으로 부터의 거리 [j] 에 따른 속도, 압력의 예측값
    output_shape = (213,141,61)
    total_spatial_error = []
    train_pos_unnorm = np.concatenate([train_data['pos'][:,i:i+1]*all_params["domain"]["in_max"][0,i]
                                 for i in range(4)],1).reshape(-1,4)
    valid_pos_unnorm = np.concatenate([valid_data['pos'][:,i:i+1]*all_params["domain"]["in_max"][0,i] 
                                 for i in range(4)],1).reshape((-1,)+output_shape+(4,))
    
    train_pos_from_center = train_pos_unnorm - np.array([0,0.028,0.006,0.00212]).reshape(-1,4)
    train_pos_from_center = np.sqrt(train_pos_from_center[:,1]**2+train_pos_from_center[:,2]**2+train_pos_from_center[:,3]**2)
    train_pos_unnorm = train_pos_unnorm.reshape(-1,4)

    valid_pos_from_center = valid_pos_unnorm[1,:,:,:,:].reshape(-1,4) - np.array([0,0.028,0.006,0.00212]).reshape(-1,4)
    valid_pos_from_center = np.sqrt(valid_pos_from_center[:,1]**2+valid_pos_from_center[:,2]**2+valid_pos_from_center[:,3]**2)
    valid_pos_unnorm = valid_pos_unnorm.reshape(-1,4)

    counts, bins, bars = plt.hist(train_pos_from_center, bins=50)

    train_indexes = []
    for i in range(bins.shape[0]-1):
        index = np.where((train_pos_from_center<bins[i+1])&(train_pos_from_center>=bins[i]))
        train_indexes.append(index[0])

    valid_indexes = []
    for i in range(bins.shape[0]-1):
        index = np.where((valid_pos_from_center<bins[i+1])&(valid_pos_from_center>=bins[i]))
        valid_indexes.append(index[0])

#%%
    train_vel_sub_list = []
    train_pos_sub_list = []
    train_pos_sub_unnorm_list = []
    for i in range(len(train_indexes)):
        train_vel_sub_list.append(train_data['vel'][train_indexes[i],:])
        train_pos_sub_unnorm_list.append(train_pos_unnorm[train_indexes[i],:])
        train_pos_sub_list.append(train_data['pos'][train_indexes[i],:])

#%%
    train_pred_list_total = []
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    for i in range(len(train_pos_sub_list)):
        pred_list = []
        print(i)
        for j in range(train_pos_sub_list[i].shape[0]//10000+1):
            pred = model_fn(all_params, train_pos_sub_list[i][10000*j:10000*(j+1),:])
            pred_list.append(pred)
        pred_list = np.concatenate(pred_list,0)
        pred_unnorm = np.concatenate([pred_list[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
        pred_unnorm[:,-1] = 1.185*pred_unnorm[:,-1]
        train_pred_list_total.append(pred_unnorm)
#%%
    train_vel_error_list = []
    for i in range(len(train_pred_list_total)):
        vel_error_list = []
        print(i)
        vel_error_list.append(np.sqrt((np.sqrt(train_pred_list_total[i][:,0]**2+train_pred_list_total[i][:,1]**2+train_pred_list_total[i][:,2]**2)-
                                       np.sqrt(train_vel_sub_list[i][:,0]**2+train_vel_sub_list[i][:,1]**2+train_vel_sub_list[i][:,2]**2))**2)/
                                       np.sqrt(train_vel_sub_list[i][:,0]**2+train_vel_sub_list[i][:,1]**2+train_vel_sub_list[i][:,2]**2))
        train_vel_error_list.append(vel_error_list)

#%%
    
    dist = getattr(st,"norm")
    train_vel_mean_error_list = []
    for i in range(len(train_vel_error_list)):
        mean_std = dist.fit(train_vel_error_list[i])
        train_vel_mean_error_list.append(mean_std[0])
    train_vel_mean_error_list = np.array(train_vel_mean_error_list)

#%%
    valid_vel_sub_t_list = []
    valid_pos_sub_t_list = []
    valid_pos_sub_t_unnorm_list = []
    for j in range(50):
        vel_sub_list = []
        pos_sub_list = []
        pos_sub_unnorm_list = []
        print(j)
        for i in range(len(valid_indexes)):
            #valid_data['vel'][:,3:4] = valid_data['vel'][:,3:4]*1.185
            vel_sub_list.append(valid_data['vel'][213*141*61*j:213*141*61*(j+1),:][valid_indexes[i],:])
            pos_sub_unnorm_list.append(valid_pos_unnorm[213*141*61*j:213*141*61*(j+1),:][valid_indexes[i],:])
            pos_sub_list.append(valid_data['pos'][213*141*61*j:213*141*61*(j+1),:][valid_indexes[i],:])
        valid_vel_sub_t_list.append(vel_sub_list)
        valid_pos_sub_t_list.append(pos_sub_list)
        valid_pos_sub_t_unnorm_list.append(pos_sub_unnorm_list)

    valid_ext_p_list = []
    for i in range(len(valid_vel_sub_t_list)):
        ext_p_array = np.mean(np.concatenate(valid_vel_sub_t_list[i],0)[:,3])
        valid_ext_p_list.append(ext_p_array)
    for i in range(len(valid_vel_sub_t_list)):
        for j in range(len(valid_vel_sub_t_list[i])):
            valid_vel_sub_t_list[i][j][:,3] = valid_vel_sub_t_list[i][j][:,3] - valid_ext_p_list[i]

    valid_pred_list_total = []
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    for i in range(len(valid_pos_sub_t_list)):
        pred_list = []
        print(i)
        for j in range(len(valid_indexes)):
            valid_pos_unnorm = np.concatenate(valid_pos_sub_t_list[i][j])
            pred = model_fn(all_params, valid_pos_sub_t_list[i][j])
            pred_unnorm = np.concatenate([pred[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
            pred_unnorm[:,-1] = 1.185*pred_unnorm[:,-1]
            pred_list.append(pred_unnorm)
        valid_pred_list_total.append(pred_list)
    valid_test_p_list = []
    for i in range(len(valid_pred_list_total)):
        test_p_array = np.mean(np.concatenate(valid_pred_list_total[i],0)[:,-1])
        valid_test_p_list.append(test_p_array)
    for i in range(len(valid_pred_list_total)):
        for j in range(len(valid_pred_list_total[i])):
            valid_pred_list_total[i][j][:,-1] = valid_pred_list_total[i][j][:,-1] - valid_test_p_list[i]

    valid_vel_error_t_list = []
    valid_pre_error_t_list = []
    for i in range(len(valid_pred_list_total)):
        vel_error_list = []
        pre_error_list = []
        print(i)
        for j in range(len(valid_pred_list_total[i])):
            vel_error_list.append(np.sqrt((np.sqrt(valid_pred_list_total[i][j][:,0]**2+valid_pred_list_total[i][j][:,1]**2+valid_pred_list_total[i][j][:,2]**2)-np.sqrt(valid_vel_sub_t_list[i][j][:,0]**2+valid_vel_sub_t_list[i][j][:,1]**2+valid_vel_sub_t_list[i][j][:,2]**2))**2)/np.sqrt(valid_vel_sub_t_list[i][j][:,0]**2+valid_vel_sub_t_list[i][j][:,1]**2+valid_vel_sub_t_list[i][j][:,2]**2))
            pre_error_list.append(np.sqrt((np.sqrt(valid_pred_list_total[i][j][:,3]**2)-np.sqrt(valid_vel_sub_t_list[i][j][:,3]**2))**2))
        valid_vel_error_t_list.append(vel_error_list)
        valid_pre_error_t_list.append(pre_error_list)
#%%
    dist = getattr(st,"norm")
    valid_vel_mean_error_list = []
    for i in range(len(valid_vel_error_t_list[10])):
        mean_std = dist.fit(valid_vel_error_t_list[10][i])
        valid_vel_mean_error_list.append(mean_std[0])
    valid_vel_mean_error_list = np.array(valid_vel_mean_error_list)
#%%
    dist = getattr(st,"norm")
    valid_pre_mean_error_list = []
    for i in range(len(valid_pre_error_t_list[10])):
        mean_std = dist.fit(valid_pre_error_t_list[10][i])
        valid_pre_mean_error_list.append(mean_std[0])
    valid_pre_mean_error_list = np.array(valid_pre_mean_error_list)
#%%
    valid_mean_error = np.concatenate([valid_vel_mean_error_list.reshape(-1,1),
                                        valid_pre_mean_error_list.reshape(-1,1)],1)
#%%
    test_vels = np.concatenate(valid_pred_list_total[25])
    test_vels_ext = np.concatenate(valid_vel_sub_t_list[25])
#%%
    f = np.concatenate([(test_vels[:,0]-test_vels_ext[:,0]).reshape(-1,1),
                        (test_vels[:,1]-test_vels_ext[:,1]).reshape(-1,1),
                        (test_vels[:,2]-test_vels_ext[:,2]).reshape(-1,1)],1)
    div = np.concatenate([test_vels_ext[:,0].reshape(-1,1), test_vels_ext[:,1].reshape(-1,1),
                          test_vels_ext[:,2].reshape(-1,1)],1)
    print(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))
    print(np.linalg.norm(test_vels[:,3]-test_vels_ext[:,3])/np.linalg.norm(test_vels_ext[:,3]))
#%%
with open("datas/"+checkpoint_fol+"/train_error_from_center.pkl","wb") as f:
    pickle.dump(train_vel_mean_error_list,f)
f.close()
with open("datas/"+checkpoint_fol+"/valid_error_from_center.pkl","wb") as f:
    pickle.dump(valid_mean_error,f)
f.close()
#%%
"""
from scipy.integrate import tplquad
L = 0.056
W = 0.012
H = 0.00424
def sphere_condition(x,y,z):
    return x**2+y**2+z**2<r**2
def intergrand(x,y,z):
    return 1 if sphere_condition(x,y,z) else 0
volumes = []
for i in range(len(bins)):
    print(i)
    r = bins[i]
    volume, error = tplquad(intergrand, -L/2, L/2, lambda x:-W/2, lambda x:W/2, lambda x,y:-H/2, lambda x,y:H/2)
    volumes.append(volume)
    sub_volumes = np.array(volumes[1:])-np.array(volumes[:-1])
    c_vol_avg = np.array(counts)/np.array(sub_volumes)
    plt.hist(bins[:-1], bins,weights=c_vol_avg)
    plt.show()
with open("datas/sub_volumes.pkl","wb") as f:
    pickle.dump(sub_volumes,f)
f.close()
with open("datas/counts.pkl","wb") as f:
    pickle.dump(counts,f)
f.close()
"""

#%%


