import torch
from FLAlgorithms.users.hierarchicaluseravg import hier_fedavg_user
from utils.model_utils import read_data, read_user_data
import copy
import os
import numpy as np
from datetime import date
import h5py
from tqdm import trange
# Implementation for pFedMe-multi Server

class hier_fedavg():
    def __init__(self,
                device,
                dataset, 
                algorithm, 
                model,
                model_name,
                batch_size,
                alpha,
                num_glob_iters, 
                local_iters, 
                team_iters,
                optimizer,
                num_users,
                tot_users,
                num_labels,
                num_teams,
                group_division, 
                exp_no):
        
        self.model_name = model_name
        self.old_global_model = copy.deepcopy(model)
        self.model = copy.deepcopy(model)  # initialize global model
        
        self.team = []
        for i in range(0,num_teams):
            self.team.append(copy.deepcopy(model))
        
        self.users = [[] for _ in range(num_teams)]
        
        self.global_train_acc = []
        self.global_test_acc = [] 
        self.global_train_loss = []
        self.global_test_loss = [] 

        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.alpha = alpha  # learning rate
        self.total_train_samples = 0
        self.user_subset = num_users
       
        # global iteration, team iterations, local iterations
        self.num_glob_iters = num_glob_iters
        self.team_iters = team_iters
        self.local_iters = local_iters
    
        
        self.algorithm = algorithm
        self.num_teams = num_teams
        self.exp_no = exp_no
        
        """           
          data[0] has clients
          data[1] has clients in 2 different groups.
          data[2] training data of users
          data[3] test data of users

        """
        
        data = read_data(dataset, tot_users, num_labels, num_teams, group_division)

        # this function divide the dataset into non-iid manner in to n clients
        # total_users = len(data[1][0])
        self.groups = data[1]
        self.model_initializer()

        for grp in range(len(data[1])):
            total_users = len(data[1][grp])
            for i in range(total_users):
                print(data[1][grp][i])
                id, train, test = read_user_data(data[1][grp][i], data, dataset)
                print("id :", id)
                user = hier_fedavg_user(device,
                                        id, 
                                        train, 
                                        test, 
                                        model, 
                                        model_name,
                                        batch_size, 
                                        alpha,
                                        local_iters,
                                        dataset )
                
                self.users[grp].append(user)
                self.total_train_samples += user.train_samples
            print("Number of users in group [", grp, "] / total users:", total_users, " / ", len(data[0]))
        print("Finished creating user server and team objects")

    
        
    def model_initializer(self):
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)

    
    def send_parameters(self, grp):
        assert (self.users[grp] is not None and len(self.users[grp]) > 0)
        # print(type(self.model))
        for user in self.users[grp]:
            # print(user)
            user.set_parameters(self.team[grp])
         
    def add_parameters(self, user, ratio, grp):

        for theta_bar_param, user_param in zip(self.team[grp].parameters(), user.get_parameters()):
            theta_bar_param.data = theta_bar_param.data + user_param.data * ratio

    
    def set_old_global_parameter(self):
        for old_param, global_param in zip(self.old_global_model.parameters(), self.model.parameters()):
            old_param.data = global_param.data.clone()

    def set_team_parameter(self, i):  # w^(t,0) = x^t
        for team_param, global_param in zip(self.team[i].parameters(), self.old_global_model.parameters()):
            team_param.data = global_param.data.clone()

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users, grp):
        if num_users == len(self.users[grp]):
            return self.users[grp]
        elif  num_users < len(self.users[grp]):
         
            num_users = min(num_users, len(self.users[grp]))
            np.random.seed(round)
            return np.random.choice(self.users[grp], num_users, replace=False)  # , p=pk)

        else: 
            assert (self.selected_users > len(self.users))
            # print("number of selected users are greater than total users")


    def aggregate_parameters(self, grp):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.team[grp].parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            # self.add_parameters(user, user.train_samples / total_train)
            self.add_parameters(user, 1 / self.user_subset, grp)


    def global_update(self):
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        ratio = float(1/self.num_teams)
        print(ratio)
        for grp in range(len(self.team)):
            for param, team_param in zip(self.model.parameters(), self.team[grp].parameters()):
                param.data = param.data + ratio * team_param.data


    # Save loss, accurancy to h5 fiel
    def save_results(self):
        today = date.today()
        print("exp_no ", self.exp_no)
        alg = str(self.dataset) + "_" + str(self.algorithm) + "_" + str(self.model_name) + "_" + "exp_no_" + str(self.exp_no) + "_" + "alpha_" + str(self.alpha) + "_" + "num_teams_" + str(self.num_teams) + "_" +  str(self.batch_size) + "b" + "_" + str(self.num_glob_iters) + "GE" + "_" + str(self.team_iters) + "TE" + "_" + str(self.local_iters) + "LE"
        print(alg)
        directory_name = self.algorithm + "/" + self.dataset + "/" + str(self.model_name) + "/" + str(self.num_teams)
        if not os.path.exists("./results/"+directory_name):
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg, self.local_iters), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('alpha', data=self.alpha) 
            hf.create_dataset('global_rounds', data=self.num_glob_iters)
            hf.create_dataset('team_rounds', data=self.team_iters)
            hf.create_dataset('local_rounds', data=self.local_iters)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.close()
    """
    Global model evaluation

    """

    def test_server(self):
        num_samples = []
        tot_correct = []
        losses = []
        user = []

        for grp in range(self.num_teams):
            user += self.users[grp]
        
        for c in user:
            ct, ls,  ns = c.test(self.model.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(ls)
            
        return num_samples, tot_correct, losses

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for grp in range(self.num_teams):
            for c in self.users[grp]:
                ct, cl, ns = c.train_error_and_loss(self.model.parameters())
                # print("ct =",ct)
                # print("cl =",cl)
                # print("ns =",ns)
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(cl * 1.0)

        
        return num_samples, tot_correct, losses

    
    def evaluate(self):

        stats_test = self.test_server()
        stats_train = self.train_error_and_loss()
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])

        self.global_train_acc.append(train_acc)
        self.global_test_acc.append(test_acc)
        self.global_train_loss.append(train_loss)
        self.global_test_loss.append(test_loss)

        print("Global Trainning Accurancy: ", train_acc)
        print("Global Trainning Loss: ", train_loss)
        print("Global test accurancy: ", test_acc)
        print("Global test_loss:",test_loss)
    

    def train(self):

        for iters in trange(self.num_glob_iters):  # t = 1.... T-1
            self.set_old_global_parameter()    # x^(t-1) = x^t
            for grp in range(self.num_teams): 
                self.set_team_parameter(grp)
                
                for team_iter in range(self.team_iters):  # k = 1 .. K-1
                    # send all parameter to the users of the group

                    self.send_parameters(grp)

                    self.selected_users = self.select_users(team_iter, self.user_subset, grp)
                    for user in self.selected_users:
                        user.train()

                    self.aggregate_parameters(grp)
                
            self.global_update()
                
            self.evaluate()
        self.save_results()
                    