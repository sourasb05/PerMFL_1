import torch
import os
import h5py
from FLAlgorithms.users.useravg import UserAVG
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
from datetime import date
from tqdm import trange
# Implementation for FedAvg Server

class FedAvg():
    def __init__(self,
                device,
                dataset,
                algorithm,
                model,
                model_name,
                batch_size,
                learning_rate,
                num_glob_iters,
                local_iters,
                optimizer,
                num_users,
                tot_users,
                num_labels,
                num_teams,
                group_division,
                times):
                
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_iters = local_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.global_model = copy.deepcopy(model)
        self.global_model_name = model_name
        self.num_users = num_users   #selected users
        self.algorithm = algorithm
        self.tot_users = tot_users
        self.num_labels = num_labels
        self.num_teams = num_teams
        self.group_division = group_division
        self.exp_no = times
        self.users = []
        self.selected_users = []
        self.global_train_acc = []
        self.global_train_loss = [] 
        self.global_test_acc = [] 
        self.global_test_loss = []
        
        data = read_data(dataset, tot_users, num_labels, num_teams, group_division)
        for i in range(tot_users):
            id, train, test = read_user_data(i, data, dataset)
            user = UserAVG(device,
                        id,
                        dataset,
                        train,
                        test,
                        model,
                        model_name,
                        batch_size,
                        learning_rate,
                        local_iters
                        )
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating FedAvg server.")

        
    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model)

    def add_parameters(self, user, ratio):
        model = self.global_model.parameters()
        for server_param, user_param in zip(self.global_model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.global_model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

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


    def train(self):
        loss = []
        for glob_iter in trange(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train()  # * user.train_samples
            self.aggregate_parameters()
           
        self.save_results()
        self.save_model()


    # Save loss, accurancy to h5 fiel
    def save_results(self):
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")
        alg = str(self.exp_no) + "_" + str(self.dataset) + "_" + str(self.algorithm) + "_" + str(self.global_model_name) + "_" + str(self.batch_size) + "b" + "_" + str(self.num_glob_iters) + "_" \
              + "_"  + d1
        
        print(alg)
       
        directory_name = self.algorithm + "/" + self.dataset + "/cl_5/" + str(self.global_model_name)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)



        with h5py.File("./results/" + directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            
            hf.close()

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
    