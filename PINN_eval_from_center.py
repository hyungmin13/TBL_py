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
#%% pred_list_total 는 모든 시간 [i] 에 대해 중심으로 부터의 거리 [j] 에 따른 속도, 압력의 예측값
    output_shape = (213,141,61)
    total_spatial_error = []
    pos_unnorm = np.concatenate([valid_data['pos'][:,i:i+1]*all_params["domain"]["in_max"][0,i] 
                                 for i in range(4)],1).reshape((-1,)+output_shape+(4,))

    pos_from_center = pos_unnorm[1,:,:,:,:].reshape(-1,4) - np.array([0,0.028,0.006,0.00212]).reshape(-1,4)
    pos_from_center = np.sqrt(pos_from_center[:,1]**2+pos_from_center[:,2]**2+pos_from_center[:,3]**2)
    pos_unnorm = pos_unnorm.reshape(-1,4)
    counts, bins, bars = plt.hist(pos_from_center, bins=50)

    indexes = []
    for i in range(bins.shape[0]-1):
        index = np.where((pos_from_center<bins[i+1])&(pos_from_center>=bins[i]))
        indexes.append(index[0])

    vel_sub_t_list = []
    pos_sub_t_list = []
    
    pos_sub_t_unnorm_list = []
    for j in range(50):
        vel_sub_list = []
        pos_sub_list = []
        pos_sub_unnorm_list = []
        for i in range(len(indexes)):
            vel_sub_list.append(valid_data['vel'][213*141*61*j:213*141*61*(j+1),:][indexes[i],:])
            pos_sub_unnorm_list.append(pos_unnorm[213*141*61*j:213*141*61*(j+1),:][indexes[i],:])
            pos_sub_list.append(valid_data['pos'][213*141*61*j:213*141*61*(j+1),:][indexes[i],:])
        vel_sub_t_list.append(vel_sub_list)
        pos_sub_t_list.append(pos_sub_list)
        pos_sub_t_unnorm_list.append(pos_sub_unnorm_list)

    pred_list_total = []
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    for i in range(len(pos_sub_t_list)):
        pred_list = []
        print(i)
        for j in range(len(indexes)):
            pos_unnorm = np.concatenate(pos_sub_t_list[i][j])
            pred = model_fn(all_params, pos_sub_t_list[i][j])
            pred_unnorm = np.concatenate([pred[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
            pred_unnorm[:,-1] = 1.185*pred_unnorm[:,-1]
            pred_list.append(pred_unnorm)
        pred_list_total.append(pred_list)
    test_p_list = []
    for i in range(len(pred_list_total)):
        test_p_array = np.mean(np.concatenate(pred_list_total[i],0)[:,-1])
        test_p_list.append(test_p_array)
    for i in range(len(pred_list_total)):
        for j in range(len(pred_list_total[i])):
            pred_list_total[i][j][:,-1] = pred_list_total[i][j][:,-1] - test_p_list[i]
    
    with open("datas/pred_list_"+checkpoint_fol+".pkl","wb") as f:
        pickle.dump(pred_list_total,f)
    f.close()
#%%
    vel_error_t_list = []
    pre_error_t_list = []
    dist = getattr(st,"norm")
    for i in range(len(pred_list_total)):
        vel_error_list = []
        pre_error_list = []
        print(i)
        for j in range(len(pred_list_total[i])):
            vel_error_list.append(np.sqrt((np.sqrt(pred_list_total[i][j][:,0]**2+pred_list_total[i][j][:,1]**2+pred_list_total[i][j][:,2]**2)-np.sqrt(vel_sub_t_list[i][j][:,0]**2+vel_sub_t_list[i][j][:,1]**2+vel_sub_t_list[i][j][:,2]**2))**2)/np.sqrt(vel_sub_t_list[i][j][:,0]**2+vel_sub_t_list[i][j][:,1]**2+vel_sub_t_list[i][j][:,2]**2))
            pre_error_list.append(np.sqrt((np.sqrt(pred_list_total[i][j][:,3]**2)-np.sqrt(vel_sub_t_list[i][j][:,3]**2))**2))
        vel_error_t_list.append(vel_error_list)
        pre_error_t_list.append(pre_error_list)

    import scipy.stats as st
    dist = getattr(st,"norm")
    mean_error_vel = []
    for i in range(len(vel_error_t_list[10])):
        mean_std = dist.fit(vel_error_t_list[10][i])
        mean_error_vel.append(mean_std[0])
    mean_error_vel = np.array(mean_error_vel)

    import scipy.stats as st
    dist = getattr(st,"norm")
    mean_error_pre = []
    for i in range(len(pre_error_t_list[10])):
        mean_std = dist.fit(pre_error_t_list[10][i])
        mean_error_pre.append(mean_std[0])
    mean_error_pre = np.array(mean_error_pre)

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
    test_vels = np.concatenate(pred_list_total[25])
    test_vels_ext = np.concatenate(vel_sub_t_list[25])
#%%
    f = np.concatenate([(test_vels[:,0]-test_vels_ext[:,0]).reshape(-1,1),
                        (test_vels[:,1]-test_vels_ext[:,1]).reshape(-1,1),
                        (test_vels[:,2]-test_vels_ext[:,2]).reshape(-1,1)],1)
    div = np.concatenate([test_vels_ext[:,0].reshape(-1,1), test_vels_ext[:,1].reshape(-1,1),
                          test_vels_ext[:,2].reshape(-1,1)],1)
    print(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))
    print(np.linalg.norm(test_vels[:,3]-test_vels_ext[:,3])/np.linalg.norm(test_vels_ext[:,3]))

