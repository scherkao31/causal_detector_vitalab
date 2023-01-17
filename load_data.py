import torch
import pickle
import numpy as np

class Dataset(torch.utils.data.Dataset):
    'Dataset containing : trajectories of a pair of agents, factual and counterfactual scene (X_agents), the label (y) and the counterfactual ADE of the egos agent trajectory when removing a certain agent (ade)'
    def __init__(self, trajs, trajs_agents, labels, ades, ic_labels):
        'Initialization'
        self.labels = labels
        self.trajs = trajs
        self.trajs_agents = trajs_agents
        self.ades = ades
        self.ic_labels = ic_labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.trajs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.trajs[index]
        X_agents = self.trajs_agents[index]
        y = torch.tensor(self.labels[index])
        ade = torch.tensor(self.ades[index])
        ic = torch.tensor(self.ic_labels[index])

        return X.float(), X_agents.float(), y, ade, ic
    
    
    
def load_dataset(nc_threshold, c_threshold, batch_size = 64, shuffle = True, file_name = 'synth_v1.a.filtered.pkl', num_agents_min = 0):

    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': shuffle}

    DATASET_PATH= file_name
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    scenes = dataset["scenes"]
    total = len(scenes)
    train_traj = scenes[:int(total * 0.8)]
    test_traj = scenes[int(total * 0.8):]
    all_causal = []

    num_agents = num_agents_min

    num = 0
    for traj in train_traj:
        scene_traj = traj['trajectories']
        agents_num = scene_traj.shape[0]
        if agents_num == num_agents:
            num += 1
    num


    ic_threshold = c_threshold



    # COMPUTE HOW MUCH OF CAUSAL/NON-CAUSAL AGENT IN THE TRAINING SET

    nb_c = 0
    nb_ic = 0
    nb_nc = 0

    for traj in train_traj:
        scene_traj = traj['trajectories']
        agents_num = scene_traj.shape[0]


        ego_traj = torch.tensor(scene_traj[0])
        ego_ade = traj['remove_agent_i_ade'][1:, 0]
        ego_label = torch.where(torch.tensor(ego_ade) == 0.0, 0, 1)
        to_check = torch.where(ego_label == 1)[0]

        if agents_num > num_agents:

            for i in range(1, agents_num):

                causality = traj['remove_agent_i_ade'][:, 0][i]
                if causality < nc_threshold or causality > c_threshold:
                    if causality < nc_threshold:
                        lab = -1.
                        for j in to_check:
                            ade = traj['remove_agent_i_ade'][i, j+1]
                            if ade > ic_threshold:
                                lab = 0.


                        if lab == - 1.:
                            nb_nc += 1
                        else:
                            nb_ic += 1
                    else :
                        nb_c += 1



    # CREATE TRAINING DATASET WITH TRAJECTORIES OF CAUSAL/NON-CAUSAL AGENTS W.R.T THE EGO
    ## The dataset set is balanced between causal and non-causal pairs
    ### the labels are dependant on the causal thresholds, and we skip situation in between those thresholds

    max_nc = nb_nc
    max_ic = nb_ic
    max_c = nb_nc + nb_ic


    all_causal = []
    trajs = []
    trajs_agents = [] 
    nb_c = 0
    nb_ic = 0
    nb_nc = 0
    ades = []
    i_causal = []

    for traj in train_traj:
        scene_traj = traj['trajectories']
        agents_num = scene_traj.shape[0]


        ego_traj = torch.tensor(scene_traj[0])
        ego_ade = traj['remove_agent_i_ade'][1:, 0]
        ego_label = torch.where(torch.tensor(ego_ade) == 0.0, 0, 1)
        to_check = torch.where(ego_label == 1)[0]

        if agents_num > num_agents:

            for i in range(1, agents_num):

                causality = traj['remove_agent_i_ade'][:, 0][i]
                if causality < nc_threshold or causality > c_threshold:
                    # Pair of trajectories : ego and considered agent
                    agent_traj = torch.tensor(scene_traj[i])
                    concat1 = torch.cat((ego_traj, agent_traj)).reshape(-1)
                    

                    diff_num_agents = 12 - agents_num
                    zeros_fill = torch.zeros(diff_num_agents, 20, 2)
            
                    traj_ = torch.tensor(scene_traj)
                    zeros_ = torch.zeros(1, 20, 2)
                    
                    # Factual and counterfactual scene : we replace by all zeros the trajectory of the considered agent

                    ego_and_agent_traj_f = torch.cat((traj_[0:1], traj_[i: i + 1]), dim = 0)
                    ego_and_agent_traj_cf = torch.cat((traj_[0:1], zeros_), dim = 0) 


                    other_traj = torch.cat((traj_[1:i], traj_[i + 1:]))


                    factual = torch.cat((ego_and_agent_traj_f, other_traj, zeros_fill), dim = 0)  
                    counterfactual = torch.cat((ego_and_agent_traj_cf, other_traj, zeros_fill), dim = 0) 

                    concat = torch.stack((factual, counterfactual))

                    if causality < nc_threshold :
                        lab = -1.
                        for j in to_check:
                            ade = traj['remove_agent_i_ade'][i, j+1]
                            if ade > ic_threshold:
                                lab = 0.


                        if lab == - 1. and nb_nc < max_nc:
                            nb_nc += 1
                            i_causal += [0.]
                            trajs += [concat1]
                            trajs_agents += [concat]
                            all_causal += [0.]
                            ades += [causality]
                        else:
                            if nb_ic < max_ic:
                                nb_ic += 1
                                i_causal += [1.]
                                trajs += [concat1]
                                trajs_agents += [concat]
                                all_causal += [0.]
                                ades += [causality]
                    else :
                        if nb_c < max_c:
                            trajs += [concat1]
                            trajs_agents += [concat]
                            all_causal += [1.]
                            ades += [causality]
                            nb_c += 1
                            i_causal += [0.]






    # COMPUTE HOW MUCH OF CAUSAL/NON-CAUSAL AGENT IN THE TRAINING SET

    nb_c = 0
    nb_ic = 0
    nb_nc = 0

    for traj in test_traj:
        scene_traj = traj['trajectories']
        agents_num = scene_traj.shape[0]

        ego_traj = torch.tensor(scene_traj[0])
        ego_ade = traj['remove_agent_i_ade'][1:, 0]
        ego_label = torch.where(torch.tensor(ego_ade) == 0.0, 0, 1)
        to_check = torch.where(ego_label == 1)[0]

        if agents_num > num_agents:

            for i in range(1, agents_num):

                causality = traj['remove_agent_i_ade'][:, 0][i]
                if causality < nc_threshold or causality > c_threshold:
                    if causality < nc_threshold:
                        lab = -1.
                        for j in to_check:
                            ade = traj['remove_agent_i_ade'][i, j+1]
                            if ade > ic_threshold:
                                lab = 0.


                        if lab == - 1.:
                            nb_nc += 1
                        else:
                            nb_ic += 1
                    else :
                        nb_c += 1




    # CREATE VALIDATION DATASET WITH TRAJECTORIES OF CAUSAL/NON-CAUSAL AGENTS W.R.T THE EGO
    ## The dataset set is balanced between causal and non-causal pairs
    ### the labels are dependant on the causal thresholds, and we skip situation in between those thresholds

    max_nc = nb_nc
    max_ic = nb_ic
    max_c = nb_nc + nb_ic



    all_causal_val = []
    val_trajs = []
    val_trajs_agents = [] 
    nb_c = 0
    nb_ic = 0
    nb_nc = 0
    ades_val = []
    i_causal_val = []
    for traj in test_traj:
        scene_traj = traj['trajectories']
        agents_num = scene_traj.shape[0]

        ego_traj = torch.tensor(scene_traj[0])
        ego_ade = traj['remove_agent_i_ade'][1:, 0]
        ego_label = torch.where(torch.tensor(ego_ade) == 0.0, 0, 1)
        to_check = torch.where(ego_label == 1)[0]

        if agents_num > num_agents:

            for i in range(1, agents_num):

                causality = traj['remove_agent_i_ade'][:, 0][i]
                if causality < nc_threshold or causality > c_threshold:
                    
                    # Pair of trajectories : ego and considered agent
                    agent_traj = torch.tensor(scene_traj[i])
                    concat1 = torch.cat((ego_traj, agent_traj)).reshape(-1)

                    diff_num_agents = 12 - agents_num
                    zeros_fill = torch.zeros(diff_num_agents, 20, 2)

                    traj_ = torch.tensor(scene_traj)
                    zeros_ = torch.zeros(1, 20, 2)
                    
                    # Factual and counterfactual scene : we replace by all zeros the trajectory of the considered agent
                    
                    ego_and_agent_traj_f = torch.cat((traj_[0:1], traj_[i: i + 1]), dim = 0)
                    ego_and_agent_traj_cf = torch.cat((traj_[0:1], zeros_), dim = 0)  

                    other_traj = torch.cat((traj_[1:i], traj_[i + 1:]))

                    factual = torch.cat((ego_and_agent_traj_f, other_traj, zeros_fill), dim = 0)  
                    counterfactual = torch.cat((ego_and_agent_traj_cf, other_traj, zeros_fill), dim = 0)  

                    concat = torch.stack((factual, counterfactual))
                    
                    if causality < nc_threshold:
                        lab = -1.
                        for j in to_check:
                            #print(i, j, to_check)
                            ade = traj['remove_agent_i_ade'][i, j+1]
                            if ade > ic_threshold:
                                lab = 0.


                        if lab == - 1 and nb_nc < max_nc:
                            nb_nc += 1
                            i_causal_val += [0.]
                            val_trajs += [concat1]
                            val_trajs_agents += [concat]
                            all_causal_val += [0.]
                            ades_val += [causality]
                        else:
                            if nb_ic < max_ic:
                                nb_ic += 1
                                i_causal_val += [1.]
                                val_trajs += [concat1]
                                val_trajs_agents += [concat]
                                all_causal_val += [0.]
                                ades_val += [causality]
                    else :
                        if nb_c < max_c:
                            val_trajs += [concat1]
                            val_trajs_agents += [concat]
                            all_causal_val += [1.]
                            ades_val += [causality]
                            nb_c += 1
                            i_causal_val += [0.]





    torch.set_default_dtype(torch.float32)
    training_set = Dataset(trajs, trajs_agents, all_causal, ades, i_causal)
    train_loader = torch.utils.data.DataLoader(training_set, **params)
    validation_set = Dataset(val_trajs, val_trajs_agents, all_causal_val, ades_val, i_causal_val)
    val_loader = torch.utils.data.DataLoader(validation_set, **params)
    
    return train_loader, val_loader


class Dataset_(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, trajs_agents, labels):
        'Initialization'
        self.labels = labels
        self.trajs = trajs
        self.trajs_agents = trajs_agents
        #self.ades = ades
        #self.ic_labels = ic_labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.trajs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.trajs[index]
        X_agents = self.trajs_agents[index]
        y = torch.tensor(self.labels[index])

        return X.float(), X_agents.float(), y

    
def load_indirectly_causal_set(orca_causal_time_steps_max = 0, nc_threshold = 0.001, c_threshold = 0.1, batch_size = 64, shuffle = True, file_name = 'synth_v1.a.filtered.pkl', num_agents_min = 0):
    
    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': shuffle}
    
    DATASET_PATH= file_name
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    scenes = dataset["scenes"]
    total = len(scenes)
    train_traj = scenes[:int(total * 0.8)]
    test_traj = scenes[int(total * 0.8):]
    
    ic_threshold = c_threshold
    
    num_agents = num_agents_min
    
    all_causal_2 = []
    val_trajs_2 = []
    val_trajs_agents_2 = [] 
    ades_val_2 = []

    for traj in test_traj:
        scene_traj = traj['trajectories']
        agents_num = scene_traj.shape[0]

        ego_traj = torch.tensor(scene_traj[0])
        ego_ade = traj['remove_agent_i_ade'][1:, 0]
        ego_label = torch.where(torch.tensor(ego_ade) == 0.0, 0, 1)
        to_check = torch.where(ego_label == 1)[0]

        if agents_num > num_agents:

            for i in range(1, agents_num):

                causality = traj['remove_agent_i_ade'][:, 0][i]
                causality_orca = traj['causality_labels'][0][:, i].sum()
                if causality > c_threshold and causality_orca <= orca_causal_time_steps_max:
                    # For MLP
                    agent_traj = torch.tensor(scene_traj[i])
                    concat1 = torch.cat((ego_traj, agent_traj)).reshape(-1)

                    diff_num_agents = 12 - agents_num
                    zeros_fill = torch.zeros(diff_num_agents, 20, 2)


                    traj_ = torch.tensor(scene_traj)
                    zeros_ = torch.zeros(1, 20, 2)

                    ego_and_agent_traj_f = torch.cat((traj_[0:1], traj_[i: i + 1]), dim = 0)
                    ego_and_agent_traj_cf = torch.cat((traj_[0:1], zeros_), dim = 0)  

                    other_traj = torch.cat((traj_[1:i], traj_[i + 1:]))
                    
                    factual = torch.cat((ego_and_agent_traj_f, other_traj, zeros_fill), dim = 0)  
                    counterfactual = torch.cat((ego_and_agent_traj_cf, other_traj, zeros_fill), dim = 0)  

                    concat = torch.stack((factual, counterfactual))

                    val_trajs_2 += [concat1]
                    val_trajs_agents_2 += [concat]
                    all_causal_2 += [1.]
                    ades_val_2 += [causality]


    val_set_ic = Dataset_(val_trajs_2, val_trajs_agents_2, all_causal_2)
    val_loader_ic = torch.utils.data.DataLoader(val_set_ic, **params)
    
    return val_loader_ic


def load_indirectly_non_causal_set(orca_causal_time_steps_min = 15, nc_threshold = 0.001, c_threshold = 0.1, batch_size = 64, shuffle = True, file_name = 'synth_v1.a.filtered.pkl', num_agents_min = 0):
    
    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': shuffle}
    
    DATASET_PATH= file_name
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    scenes = dataset["scenes"]
    total = len(scenes)
    train_traj = scenes[:int(total * 0.8)]
    test_traj = scenes[int(total * 0.8):]
    
    ic_threshold = c_threshold
    
    num_agents = num_agents_min
    
    all_causal_2 = []
    val_trajs_2 = []
    val_trajs_agents_2 = [] 
    ades_val_2 = []

    for traj in test_traj:
        scene_traj = traj['trajectories']
        agents_num = scene_traj.shape[0]

        ego_traj = torch.tensor(scene_traj[0])
        ego_ade = traj['remove_agent_i_ade'][1:, 0]
        ego_label = torch.where(torch.tensor(ego_ade) == 0.0, 0, 1)
        to_check = torch.where(ego_label == 1)[0]

        if agents_num > num_agents:

            for i in range(1, agents_num):

                causality = traj['remove_agent_i_ade'][:, 0][i]
                causality_orca = traj['causality_labels'][0][:, i].sum()
                if causality < nc_threshold and causality_orca >= orca_causal_time_steps_min:
                    # For MLP
                    agent_traj = torch.tensor(scene_traj[i])
                    concat1 = torch.cat((ego_traj, agent_traj)).reshape(-1)

                    diff_num_agents = 12 - agents_num
                    zeros_fill = torch.zeros(diff_num_agents, 20, 2)


                    traj_ = torch.tensor(scene_traj)
                    zeros_ = torch.zeros(1, 20, 2)

                    ego_and_agent_traj_f = torch.cat((traj_[0:1], traj_[i: i + 1]), dim = 0)
                    ego_and_agent_traj_cf = torch.cat((traj_[0:1], zeros_), dim = 0)  

                    other_traj = torch.cat((traj_[1:i], traj_[i + 1:]))
                    
                    factual = torch.cat((ego_and_agent_traj_f, other_traj, zeros_fill), dim = 0)  
                    counterfactual = torch.cat((ego_and_agent_traj_cf, other_traj, zeros_fill), dim = 0)  

                    concat = torch.stack((factual, counterfactual))

                    val_trajs_2 += [concat1]
                    val_trajs_agents_2 += [concat]
                    all_causal_2 += [0.]
                    ades_val_2 += [causality]


    val_set_inc = Dataset_(val_trajs_2, val_trajs_agents_2, all_causal_2)
    val_loader_inc = torch.utils.data.DataLoader(val_set_inc, **params)
    
    return val_loader_inc

