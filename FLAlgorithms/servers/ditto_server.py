import torch
import os
import copy
import numpy as np
from FLAlgorithms.users.ditto_users import DittoUser
from utils.model_utils import read_data, read_user_data
from tqdm import trange
import h5py
from datetime import date

class ditto_server():
    def __init__(self, device,
                  dataset, 
                  algorithm, 
                  model, 
                  model_name, 
                  batch_size, 
                  learning_rate, 
                  ditto_lambda, 
                  num_global_iters, 
                  num_local_iters,
                  total_users,
                  subset_users,
                  num_labels,
                  num_groups,
                  group_division,
                  exp_no
                  ):
        
        data = read_data(dataset, total_users, num_labels, num_groups, group_division)
        self.total_users = total_users
        self.users = []
        
        self.dataset = dataset
        self.algorithm = algorithm
        self.model_name = model_name
        self.global_iters = num_global_iters
        self.local_iters = num_local_iters
        self.batch_size = batch_size
        self.subset_users = subset_users
        self.learning_rate = learning_rate
        self.ditto_lambda = ditto_lambda



        self.exp_no = exp_no

        self.global_model = copy.deepcopy(model)

        self.device_train_acc = []
        self.device_test_acc = []
        self.device_train_loss = []
        self.device_test_loss = []

        self.global_train_acc = []
        self.global_test_acc = [] 
        self.global_train_loss = []
        self.global_test_loss = [] 

        self.avg_per_train_acc = []
        self.avg_per_train_loss = []        
        self.avg_per_test_acc = []        
        self.avg_per_test_loss = []
        
        
        for i in range(total_users):
            id, train, test = read_user_data(i, data, dataset)
            user = DittoUser(device,
                            id, 
                            train, 
                            test, 
                            model, 
                            batch_size, 
                            learning_rate,
                            ditto_lambda,
                            num_local_iters)
            self.users.append(user)
            
        print("Number of users / total users:",len(self.users), " / " ,total_users)
        print("Finished creating ditto server")


    def select_users(self, round, subset_users):
        # selects num_clients clients weighted by number of samples from possible_clients
        # self.selected_users = []
        # print("num_users :",num_users)
        # print(" size of user per group :",len(self.users[grp]))
        if subset_users == len(self.users):
            # print("All users are selected")
            # print(self.users[grp])
            return self.users
        elif  subset_users < len(self.users):
         
            np.random.seed(round)
            return np.random.choice(self.users, subset_users, replace=False)  # , p=pk)

        else: 
            assert (self.subset_users > len(self.users))
            # print("number of selected users are greater than total users")

    def model_initializer(self):
        for param in self.global_model.parameters():
            param.grad = torch.zeros_like(param.data)
    
    def send_global_parameters(self):
        for user in self.selected_users:
            user.get_parameters(self.global_model)
        
    
    def add_parameters(self, user, ratio):
        for g_param, d_tk in zip(self.global_model.parameters(), user.delta_tk.parameters()):
            g_param.data = g_param.data + torch.mul(g_param, d_tk)*ratio
    
    
    def global_update(self):
        
        assert (self.selected_users is not None and len(self.selected_users) > 0)
       
        for user in self.selected_users:
            self.add_parameters(user, float(1/self.subset_users))
        
    def save_results(self):
        today = date.today()
        # d1 = today.strftime("_%d_%m_%Y")
        print("exp_no ", self.exp_no)
        alg = self.dataset + "_" + self.algorithm + "_" + str(self.model_name) + "_exp_no_" + str(self.exp_no) + "_lr_" + str(self.learning_rate) + "_lambda_" + str(self.ditto_lambda) + "_b_" +  str(self.batch_size) +  "_GE_" + str(self.global_iters) + "_LE_" + str(self.local_iters) 
        print(alg)
       
        directory_name = self.algorithm + "/" + self.dataset + "/" + str(self.model_name)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('alpha', data=self.learning_rate) 
            hf.create_dataset('lambda', data=self.ditto_lambda)
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
        
        for c in self.users:
            ct, ls,  ns = c.test(self.global_model.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(ls)
            
        return num_samples, tot_correct, losses

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss(self.global_model.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        
        return num_samples, tot_correct, losses

    
    """
    Personalized model evaluation

    """
    def test_personalized_model(self):
        
        num_samples = []
        tot_correct = []
        tot_loss = []
        for c in self.selected_users:
            ct, ls, ns = c.test_personalized_model()
            tot_correct.append(ct * 1.0)
            tot_loss.append(ls * 1.0)
            num_samples.append(ns)

            
        return num_samples, tot_correct, tot_loss

    def train_error_and_loss_personalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.selected_users:
            ct, cl, ns = c.train_error_and_loss_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        
        return num_samples, tot_correct, losses
    
    
    def evaluate_personalized_model(self):
        stats_train = self.train_error_and_loss_personalized_model()
        stats_test = self.test_personalized_model()
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        
        
        return train_acc, train_loss, test_acc, test_loss
    

    def evaluate_global_model(self): 
        stats_test = self.test_server()
        stats_train = self.train_error_and_loss()
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])

        return train_acc, train_loss, test_acc, test_loss
    



    def evaluate(self):

        # personalized evaluation
       
        eval_per =  self.evaluate_personalized_model()

        print("Average Personal Train Accuracy: ", float(eval_per[0]))
        print("Average Personal Train Loss: ", float(eval_per[1]))
        print("Average Personal Test Accuracy: ", float(eval_per[2]))
        print("Average Personal Test Loss: ", float(eval_per[3]))
        
        self.avg_per_train_acc.append(float(eval_per[0]))
        self.avg_per_train_loss.append(float(eval_per[1]))
        self.avg_per_test_acc.append(float(eval_per[2]))
        self.avg_per_test_loss.append(float(eval_per[3]))
        
        # global model evaluation

        eval_global = self.evaluate_global_model()

        self.global_train_acc.append(eval_global[0])
        self.global_train_loss.append(eval_global[1])
        self.global_test_acc.append(eval_global[2])
        self.global_test_loss.append(eval_global[3])

        print("Global Trainning Accurancy: ", eval_global[0])
        print("Global Trainning Loss: ", eval_global[1])
        print("Global test accurancy: ", eval_global[2])
        print("Global test_loss:",eval_global[3])
       

     

    def train(self):
        loss = []
        self.model_initializer()
        for glob_iters in trange(self.global_iters):
            self.selected_users = self.select_users(glob_iters, self.subset_users)
            self.send_global_parameters()
            for user in self.selected_users:
                user.train()
                user.ditto_train(self.global_model)
                user.calc_delta_tk(self.global_model)
            
            self.global_update()
            
            self.evaluate()
        
        # self.save_model()
        
        self.save_results()
    
        

