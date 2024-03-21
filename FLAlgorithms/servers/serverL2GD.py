# import torch
from FLAlgorithms.users.userL2GD import UserL2GD
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
import os
import random
import h5py
import torch
from datetime import date
from tqdm import trange
# Implementation for pFedMT Server

class L2GD():
    def __init__(self,
                 device, 
                 dataset, 
                 algorithm, 
                 model, 
                 model_name, 
                 batch_size, 
                 eta,
                 num_glob_iters,
                 local_iters,
                 tot_users,
                 num_labels,
                 group_division,
                 num_cluster, 
                 user_subset,
                 exp_no):
        # Initialize data for all  users

        # read_data(dataset): this function divide the dataset into non-iid manner.
        # into n number of clients.
        #  returns a list of list named data.
        #  data[0] has clients
        #  data[1] has clients in 2 different groups.
        #  data[2] training data of users
        #  data[3] test data of users
        self.model_name = model_name
        self.device = device
        self.dataset = dataset
        self.global_iters = num_glob_iters
        self.local_iters = local_iters
        self.batch_size = batch_size
        self.num_cluster = num_cluster
        self.tot_user = tot_users
        self.user_subset = user_subset
        self.exp_no = exp_no
        self.eta = eta    # learning rate
        self.gamma = [[1. for _ in range(int(tot_users/num_cluster))] for _ in range(num_cluster)]
        self.lamda = [1.]*int(num_cluster)
        self.tau = [1.]*int(num_cluster)
        self.alpha = self.get_alphas()
        print("self.alpha :",self.alpha[0])
        self.algorithm = algorithm
        self.total_train_samples = 0
        self.theta_bar = copy.deepcopy(model)
        self.theta_bar_j = []
        for i in range(0,num_cluster):
            self.theta_bar_j.append(copy.deepcopy(model)) 

        

        self.users = [[] for _ in range(num_cluster)]
        
        self.p_0 = np.random.rand()
        self.p_j = np.random.rand(num_cluster)
        
        
        self.avg_train_acc, self.avg_train_loss = [], []
        self.avg_test_acc, self.avg_test_loss = [], []


        self.global_train_acc = []
        self.global_test_acc = [] 
        self.global_train_loss = []
        self.global_test_loss = [] 

        self.avg_per_train_acc = []
        self.avg_per_train_loss = []        
        self.avg_per_test_acc = []        
        self.avg_per_test_loss = []


    
        # print(dataset)
        data = read_data(dataset, tot_users, num_labels, num_cluster, group_division)

        # this function divide the dataset into non-iid manner in to n clients
        # total_users = len(data[1][0])
        
        
        self.model_initializer()
        
        for grp in range(num_cluster):
            group_users = len(data[1][grp])
            
            for i in range(group_users):
                id, train, test = read_user_data(data[1][grp][i], data, dataset)
                               
                user = UserL2GD(device,
                                id,
                                train,
                                test,
                                self.theta_bar,
                                model_name,
                                batch_size, 
                                self.alpha[grp], 
                                self.gamma[grp][i], 
                                self.tau[grp], 
                                self.p_0,  
                                self.p_j[grp], 
                                self.eta, 
                                local_iters)
                
                self.users[grp].append(user)
                
            print("Number of users in group [", grp, "]", len(self.users[grp]))
            print("Finished creating user server and team objects")


    def model_initializer(self):
        for param in self.theta_bar.parameters():
            param.grad = torch.zeros_like(param.data)

    
    def send_parameters(self, grp):
        assert (self.users[grp] is not None and len(self.users[grp]) > 0)
        for user in self.users[grp]:
            user.set_parameters(self.theta_bar)
        
    def get_alphas(self):
        res = np.zeros(len(self.lamda))
        for j in range(self.num_cluster):
            print("sum(client_gammas)",sum(np.array(self.gamma[j])))
            res[j] = self.lamda[j] / (self.lamda[j] + sum(np.array(self.gamma[j])))
        return res

    

    

    def add_parameters(self, user, grp, ratio):
        # print(self.theta_bar_j[grp])
        for theta_bar_j_param, user_param in zip(self.theta_bar_j[grp].parameters(), user.get_parameters()):
            theta_bar_j_param.data = theta_bar_j_param.data + user_param.data.clone() * ratio
            
    def cluster_avg(self, grp):
        
        for id, user in enumerate(self.users[grp]):
            self.add_parameters(user, grp, float(float(self.gamma[grp][id]) /np.sum(np.array(self.gamma[grp]))))

    def add_parameters_net_avg(self, user, ratio):
        for theta_bar_t, user_param in zip(self.theta_bar.parameters(), user.get_parameters()):
            theta_bar_t.data = theta_bar_t.data + user_param.data.clone() * ratio
        

    def network_avg(self,grp):
        denom = 0
        for c in  range(self.num_cluster):
            for i in range(self.user_subset):  
                denom += self.alpha[c]*self.gamma[c][i]
        
        for id, user in enumerate(self.users[grp]):
            self.add_parameters_net_avg(user, float((self.alpha[grp]*self.gamma[grp][id])/denom))
            
    

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
         
    # Save loss, accurancy to h5 file
    
    def save_results(self):
        today = date.today()
        # d1 = today.strftime("_%d_%m_%Y")
        print("exp_no ", self.exp_no)
        alg = self.dataset + "_" + self.algorithm + "_" + str(self.model_name) + "_exp_no_" + str(self.exp_no) + "_GE_" + str(self.global_iters) + "_LE_" + str(self.local_iters) 
        print(alg)
       
        directory_name = self.algorithm + "/" + self.dataset + "/" + str(self.model_name) + "/" + str(self.num_cluster)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('alpha', data=self.alpha) 
            hf.create_dataset('lambda', data=self.lamda)
            hf.create_dataset('gamma', data=self.gamma)
            hf.create_dataset('tau', data=self.tau)
            hf.create_dataset('global_rounds', data=self.global_iters)
            hf.create_dataset('local_rounds', data=self.local_iters)
            
            hf.create_dataset('per_train_accuracy', data=self.avg_per_train_acc)
            hf.create_dataset('per_train_loss', data=self.avg_per_train_loss)
            hf.create_dataset('per_test_accuracy', data=self.avg_per_test_acc)
            hf.create_dataset('per_test_loss', data=self.avg_per_test_loss)

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

        for grp in range(self.num_cluster):
            user += self.users[grp]
        
        for c in user:
            ct, ls,  ns = c.test(self.theta_bar.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(ls)
            
        return num_samples, tot_correct, losses

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for grp in range(self.num_cluster):
            for c in self.users[grp]:
                ct, cl, ns = c.train_error_and_loss(self.theta_bar.parameters())
                # print("ct =",ct)
                # print("cl =",cl)
                # print("ns =",ns)
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(cl * 1.0)

        
        return num_samples, tot_correct, losses

    
    """
    Personalized model evaluation

    """
    def test_personalized_model(self, grp):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        tot_loss = []
        i = 0
        for c in self.users[grp]:
            ct, ls, ns = c.test_personalized_model()
            tot_correct.append(ct * 1.0)
            tot_loss.append(ls * 1.0)
            num_samples.append(ns)

            # print("user id : ", i, "accuracy :", ct / ns)
            i += 1
        # ids = [c.id for c in self.users[grp]]

        return num_samples, tot_correct, tot_loss

    def train_error_and_loss_personalized_model(self, grp):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users[grp]:
            ct, cl, ns = c.train_error_and_loss_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)


        return num_samples, tot_correct, losses

    
    def evaluate_personalized_model(self, grp):
        stats_test = self.test_personalized_model(grp)
        stats_train = self.train_error_and_loss_personalized_model(grp)
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        
        return train_acc, train_loss, test_acc, test_loss


    """
    Global model evaluation

    """
    def evaluate(self):

        per_device_train_acc = 0
        per_device_train_loss = 0
        per_device_test_acc = 0
        per_device_test_loss = 0

        for grp in range(self.num_cluster):
            eval_device =  self.evaluate_personalized_model(grp)
            # eval_team =  self.evaluate_personalized_team_model(grp)
            # print(eval_device)
            per_device_train_acc += float(eval_device[0])
            per_device_train_loss += float(eval_device[1])
            per_device_test_acc += float(eval_device[2])
            per_device_test_loss += float(eval_device[3])

        print("Average Personal Train Accuracy: ", per_device_train_acc/self.num_cluster)
        print("Average Personal Train Loss: ", per_device_train_loss/self.num_cluster)
        print("Average Personal Test Accuracy: ", per_device_test_acc/self.num_cluster)
        print("Average Personal Test Loss: ", per_device_test_loss/self.num_cluster)
        
        self.avg_per_train_acc.append(per_device_train_acc/self.num_cluster)
        self.avg_per_test_acc.append(per_device_test_acc/self.num_cluster)
        self.avg_per_train_loss.append(per_device_train_loss/self.num_cluster)
        self.avg_per_test_loss.append(per_device_test_loss/self.num_cluster)
        


        # print("stats test")
        stats_test = self.test_server()
        # print(stats)
        # print("stats train")
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
        for iters in trange(self.global_iters): # t = 1,2,... do
            
            zeta_0 =  np.random.binomial(1, self.p_0)
            print("zeta_0 :",zeta_0)
            if zeta_0 == 1:
                for grp in range(self.num_cluster):
                    self.cluster_avg(grp)   # computing self.theta_bar_j
                
                
                for grp in range(self.num_cluster):
                    self.network_avg(grp)   # computing self.theta_bar
                
                
                for grp in range(self.num_cluster):
                    for user in self.users[grp]:
                        user.compute_theta_1(self.theta_bar, self.theta_bar_j[grp])
                    
            else: 
                for grp in range(self.num_cluster):
                    
                    zeta_j =  np.random.binomial(1, self.p_j[grp])
                    print("zeta_j [",grp,"] :",zeta_j)
                    if zeta_j == 1:
                        self.cluster_avg(grp)

                        for user in self.users[grp]:
                            user.compute_theta_2(self.theta_bar_j[grp])
                            
                    else:
                        # self.selected_users = self.select_users(team_iter, self.num_users, grp)

                        for user in self.users[grp]:
                            user.local_train(self.local_iters)
                        
            self.evaluate()
        self.save_results()
        # self.save_model()