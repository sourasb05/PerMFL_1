from utils.model_utils import read_data, read_user_data
from FLAlgorithms.users import user_hqsgd
import numpy as np
import copy
import os
import random
import h5py
import torch
from datetime import date

class QSGD_server():

    def __init__(self, device,
                dataset,
                algorithm,
                model,
                batch_size,
                learning,
                global_iters,
                local_iters,
                team_iters,
                optimizer,
                user_subset,
                model_name,
                expr_no):



        # Initialize data for all  users

        # read_data(dataset): this function divide the dataset into non-iid manner.
        # into n number of clients.
        #  returns a list of list named data.
        #  data[0] has clients
        #  data[1] has clients in 2 different groups.
        #  data[2] training data of users
        #  data[3] test data of users
        self.model_name = model_name
        self.old_global_model = copy.deepcopy(model)
        self.theta_bar = copy.deepcopy(model)
        self.theta_bar1 = copy.deepcopy(model)
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)  # initialize global model
        # self.global_model = copy.deepcopy(list(self.model.parameters())) # initialize global model parameters
        self.users = [[], []]
        self.num_users = num_users

        self.times = 5
        self.num_group = 2
        self.exp_no = exp_no
        self.gamma = [1.]*int(self.num_group)
        self.lamda = [1.]*int(self.num_group)
        self.tau = [1.]*int(self.num_group)
        self.alpha = self.get_alphas()
        print("self.alpha :",self.alpha[0])
        # self.tau = np.ones((2))
        # self.lamda = np.ones((2))
        # self.gamma = np.ones((2))

        self.eta = eta    # learning rate
        self.algorithm = algorithm

        self.global_train_acc, self.global_train_loss = [], []
        self.global_test_acc, self.global_test_loss = [], []

        self.server_agg_test_acc = []

    
    
    
    
    
        data = read_data(dataset)

        # this function divide the dataset into non-iid manner in to n clients
        # total_users = len(data[1][0])
        self.p_0 = np.random.random(1)[0]
        self.p_j = []
        self.groups = data[1]
        print("groups :::", self.groups)
        print(model[0])
        self.model_initializer()
        
        for grp in range(len(data[1])):
            total_users = len(data[1][grp])
            self.p_j.append(np.random.random(1)[0])
            print("self.p_j[",grp,"]",self.p_j[grp])
            for i in range(total_users):
                # print(data[1][grp][i])
                id, train, test = read_user_data(data[1][grp][i], data, dataset)
                # print("id :", id)
                
                
                user = UserL2GD(device, id, train, test, model, batch_size, 
                self.alpha[grp], self.gamma[grp], self.tau[grp], 
                self.p_0,  self.p_j[grp], self.eta, local_epochs, optimizer)
                
                self.users[grp].append(user)
                self.total_train_samples += user.train_samples
            print("Number of users in group [", grp, "] / total users:", total_users, " / ", len(data[0]))
            print("Finished creating user server and team objects")


    def model_initializer(self):
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self, grp):
        assert (self.users[grp] is not None and len(self.users[grp]) > 0)
        # print(type(self.model))
        if grp == 0:
            for user in self.users[grp]:
                # print(user)
                user.set_parameters(self.theta_bar0)
                # explore(user)
        elif grp == 1:
            for user in self.users[grp]:
                # print(user)
                user.set_parameters(self.theta_bar1)

                # explore(user)
        else:
            print("error")
    
    def get_alphas(self):
        res = np.zeros(len(self.lamda))
        for j in range(self.num_group):
            print("sum(client_gammas)",sum(self.gamma))
            res[j] = self.lamda[j] / (self.lamda[j] + sum(self.gamma))
        return res

    def network_avg(self):
        for global_param, theta_bar0_param, theta_bar1_param in zip(self.model.parameters(), self.theta_bar0.parameters(), self.theta_bar1.parameters()):
                global_param.data = (self.alpha[0]*theta_bar0_param.data + self.alpha[1]* theta_bar1_param.data)/ sum(self.alpha)


        

    def add_parameters(self, grp, user, gamma, sum_gamma):
        if grp == 0:
            for theta_bar_param, user_param in zip(self.theta_bar0.parameters(), user.get_parameters()):
                theta_bar_param.data = theta_bar_param.data + user_param.data.clone() * gamma / sum_gamma
        elif grp == 1:
            for theta_bar_param, user_param in zip(self.theta_bar1.parameters(), user.get_parameters()):
                theta_bar_param.data = theta_bar_param.data + user_param.data.clone() * gamma / sum_gamma
            
    def cluster_avg(self, grp):
        assert (self.users is not None and len(self.users) > 0)
        if grp == 0:
            for param in self.theta_bar0.parameters():
                param.data = torch.zeros_like(param.data)
        elif grp == 1:
            for param in self.theta_bar1.parameters():
                param.data = torch.zeros_like(param.data)
        # print(self.theta_bar.parameters())
        total_train = 0
        # if(self.num_users = self.to)
        # print("line 115 :",self.selected_users)
        for user in self.users[grp]:
            
            # print("user.train_samples :",user.train_samples)
            total_train += user.train_samples
            # print(user.train_samples)
        # print(total_train)
        # input("press")
        for user in self.users[grp]:
            self.add_parameters(user, grp, self.gamma[grp], sum(self.gamma))

    def set_old_global_parameter(self):
        for old_param, global_param in zip(self.old_global_model.parameters(), self.model.parameters()):
            old_param.data = global_param.data.clone()


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
        # selects num_clients clients weighted by number of samples from possible_clients
        # self.selected_users = []
        # print("num_users :",num_users)
        # print(" size of user per group :",len(self.users[grp]))
        if num_users == len(self.users[grp]):
            # print("All users are selected")
            # print(self.users[grp])
            return self.users[grp]
        elif  num_users < len(self.users[grp]):
         
            # self.selected_users =  random.sample(self.users[grp], num_users)
            # return self.selected_users
            num_users = min(num_users, len(self.users[grp]))
            np.random.seed(round)
            return np.random.choice(self.users[grp], num_users, replace=False)  # , p=pk)

        else: 
            assert (self.selected_users > len(self.users))
            # print("number of selected users are greater than total users")
        
    
   
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")
        alg = self.dataset + "_" + self.algorithm
        print("exp_no ", self.exp_no)
        alg = alg + "_" + str(self.model_name) + "_" + "exp_no_" + str(self.exp_no) + "_" + "_alpha_" + str(self.alpha) + "_" + "_beta_" + str(
            self.beta) + "_" + "_lambda_" + str(self.lamda) + "_" + "_gamma_" + str(self.gamma) + "_" + "tau_1_"+ str(self.tau1) + "_" + "tau_2_" + str(self.tau2) + "_" +   \
              str(self.batch_size) + "b" + "_" + str(self.num_glob_iters) + "GE" + "_" + str(self.team_iters) + "TE" + \
              "_" + str(self.local_epochs) + "LE" + '_' + d1
        print(alg)
        # if self.algorithm == "pFedMe":
        #    alg = alg + "_" + str(self.alpha)
        # alg = alg + "_" + str(self.times)
        # Create a directory named "new_dir"
        directory_name = self.algorithm + "/" + self.dataset + "/" + str(self.model_name) + "/lambda" 
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
            hf.create_dataset('alpha', data=self.alpha) 
            hf.create_dataset('beta', data=self.beta)
            hf.create_dataset('gamma', data=self.gamma)
            hf.create_dataset('lambda', data=self.lamda)
            hf.create_dataset('eta', data=self.eta)
            hf.create_dataset('tau', data=self.tau1)
            hf.create_dataset('global rounds', data=self.num_glob_iters)
            hf.create_dataset('team rounds', data=self.team_iters)
            hf.create_dataset('local rounds', data=self.local_epochs)
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('device_team_0_personalized__train_accuracy', data=self.device_team_0_per_train_acc)  # 0
            hf.create_dataset('device_team_0_personalized__train_loss', data=self.device_team_0_per_train_loss)  # 1
            hf.create_dataset('device_team_0_personalized__test_accuracy', data=self.device_team_0_per_test_acc)  # 2
            hf.create_dataset('device_team_1_personalized__train_accuracy', data=self.device_team_1_per_train_acc)  # 3
            hf.create_dataset('device_team_1_personalized__train_loss', data=self.device_team_1_per_train_loss)  # 4
            hf.create_dataset('device_team_1_personalized__test_accuracy', data=self.device_team_1_per_test_acc)  # 5
            hf.create_dataset('team_0_train_accuracy', data=self.team_0_train_acc)  # 6
            hf.create_dataset('team_0_test_accuracy', data=self.team_0_test_acc)  # 7
            hf.create_dataset('team_0_train_loss', data=self.team_0_train_loss)  # 8
            hf.create_dataset('team_1_train_accuracy', data=self.team_1_train_acc)  # 9
            hf.create_dataset('team_1_test_accuracy', data=self.team_1_test_acc)  # 10
            hf.create_dataset('team_1_train_loss', data=self.team_1_train_loss)  # 11
            hf.create_dataset('client_team_0_test_accuracy', data=self.client_team_0_test_acc)  # 12
            hf.create_dataset('client_team_1_test_accuracy', data=self.client_team_1_test_acc)  # 13
            hf.create_dataset('server_aggregation_test_accuracy', data=self.server_agg_test_acc)  # 14
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)  # 15
            hf.close()

    def test_server(self):
        num_samples = []
        tot_correct = []
        losses = []
        user = []

        for grp in range(2):
            user += self.users[grp]
        #  print(user)
        for c in user:
            ct, ns = c.test(self.model.parameters())
            # print("+_+_+_+_")
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            # :w
            # print("user", c.id, "accuracy:", ((ct * 1.0)/ns)*100)
            ids = [c.id for c in user]
        # print(tot_correct)
        # print(num_samples)
        # print("accuracy :", np.sum(tot_correct) / np.sum(num_samples))
        return ids, num_samples, tot_correct

    def test_server_level_aggregation(self):
        num_samples = []
        tot_correct = []
        losses = []
        user = []

        for grp in range(2):
            user += self.users[grp]
        #  print(user)
        for c in user:
            ct, ns = c.test(self.w_bar.parameters())
            # print("+_+_+_+_")
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            # :w
            # print("user", c.id, "accuracy:", ((ct * 1.0)/ns)*100)
            ids = [c.id for c in user]
        # print(tot_correct)
        # print(num_samples)
        # print("accuracy :", np.sum(tot_correct) / np.sum(num_samples))
        return ids, num_samples, tot_correct

    def train_error_and_loss(self, grp):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users[grp]:
            ct, cl, ns = c.train_error_and_loss()
            # print("ct =",ct)
            # print("cl =",cl)
            # print("ns =",ns)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users[grp]]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_personalized_model(self, grp):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        i = 0
        for c in self.users[grp]:
            ct, ns = c.test_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

            # print("user id : ", i, "accuracy :", ct / ns)
            i += 1
        ids = [c.id for c in self.users[grp]]

        return ids, num_samples, tot_correct

    def test_thetabar(self, grp):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        i = 0
        for c in self.users[grp]:
            ct, ns = c.test_thetabar(self.theta_bar.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

            # print("user id : ", i, "accuracy :", ct / ns)
            i += 1
        ids = [c.id for c in self.users[grp]]

        return ids, num_samples, tot_correct

    def test_personalized_team_model(self, grp):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        i = 0
        for c in self.users[grp]:
            if grp == 0:
                ct, ns = c.test_personalized_team_model(self.team0_model.parameters())
            else:
                ct, ns = c.test_personalized_team_model(self.team1_model.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

            # print("user id : ", i, "accuracy :", ct / ns)
            i += 1
        ids = [c.id for c in self.users[grp]]

        return ids, num_samples, tot_correct

    def train_error_and_loss_personalized_model(self, grp):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users[grp]:
            ct, cl, ns = c.train_error_and_loss_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users[grp]]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_personalized_team_model(self, grp):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users[grp]:
            ct, cl, ns = c.train_error_and_loss_personalized_team_model(self.model.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users[grp]]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def client_level_evaluate(self, grp):
        # print("stats test")
        stats = self.test_thetabar(grp)
        thetabar_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        if grp == 0:
            self.client_team_0_test_acc.append(thetabar_acc)
        elif grp == 1:
            self.client_team_1_test_acc.append(thetabar_acc)
        else:
            print("Number of group exceeds two")

        print("client level Accurancy: ", thetabar_acc)

    def evaluate(self):
        # print("stats test")
        stats = self.test_server()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        
        self.server_agg_test_acc.append(glob_acc)
        print("Average Global Accurancy: ", glob_acc)
        
    def evaluate_server_aggregation(self):
 
        stats = self.test_server_level_aggregation()
      
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        

        self.global_test_acc.append(glob_acc)
        print("Server aggregation Test Accurancy: ", glob_acc)
       

    def evaluate_personalized_model(self, grp):
        stats = self.test_personalized_model(grp)
        stats_train = self.train_error_and_loss_personalized_model(grp)
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        # self.rs_glob_acc_per.append(glob_acc)
        # self.rs_train_acc_per.append(train_acc)
        # self.rs_train_loss_per.append(train_loss)
        """if grp == 0:
            self.device_team_0_per_train_acc.append(train_acc)
            self.device_team_0_per_train_loss.append(train_loss)
            self.device_team_0_per_test_acc.append(glob_acc)
        elif grp == 1:
            self.device_team_1_per_train_acc.append(train_acc)
            self.device_team_1_per_train_loss.append(train_loss)
            self.device_team_1_per_test_acc.append(glob_acc)
        else:
            print("error")"""

        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ", train_loss)

    def evaluate_personalized_team_model(self, grp):
        stats = self.test_personalized_team_model(grp)
        stats_train = self.train_error_and_loss_personalized_team_model(grp)
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        # self.rs_glob_acc_per.append(glob_acc)
        # self.rs_train_acc_per.append(train_acc)
        # self.rs_train_loss_per.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        if grp == 0:
            self.team_0_train_acc.append(train_acc)
            self.team_0_train_loss.append(train_loss)
            self.team_0_test_acc.append(glob_acc)
        elif grp == 1:
            self.team_1_train_acc.append(train_acc)
            self.team_1_train_loss.append(train_loss)
            self.team_1_test_acc.append(glob_acc)
        else:
            print("err")
        print("Team [", grp, "] Average Personalized Accurancy: ", glob_acc)
        # print("Team [", grp, "] Average Personal Trainning Accurancy: ", train_acc)
        # print("Team [", grp, "] Average Personal Trainning Loss: ", train_loss)




  
    def train(self):
        for iters in range(self.num_glob_iters): # t = 1,2,... do
            print(" ------ Global round number: ", iters + 1, "---------")
            xi_0 =  np.random.binomial(1, self.p_0)
            print("xi_0 :",xi_0)
            if xi_0 == 1:
                for grp in range(len(self.groups)):
                    self.cluster_avg(grp)   ## computing self.theta_bar0 and self.theta_bar1
                self.network_avg()  ## computing global model "self.model"
                self.evaluate()
                for grp in range(len(self.groups)):
                    if grp == 0:
                        for user in self.users[grp]:
                            # print("[", user, "]")
                            user.compute_theta(self.model, self.theta_bar0, xi_0)
                    elif grp == 1:
                        for user in self.users[grp]:
                            # print("[", user, "]")
                            user.compute_theta(self.model, self.theta_bar1, xi_0)

            else: 
                for grp in range(len(self.groups)):
                    
                    xi_j =  np.random.binomial(1, self.p_j[grp])
                    print(print("xi_j :",xi_j))
                    if xi_j == 1:
                        self.cluster_avg(grp)

                        for user in self.users[grp]:
                            # print("[", user, "]")
                            if grp == 0:
                                user.compute_theta(self.model, self.theta_bar0, xi_0)
                            elif grp == 1:
                                user.compute_theta(self.model, self.theta_bar1, xi_0)
                            else:
                                print("Group number exceed 2")
                    else:
                        # self.selected_users = self.select_users(team_iter, self.num_users, grp)

                        for user in self.users[grp]:
                        # print("[", user, "]")
                            user.train(self.local_epochs)
                        print("group:", grp)
                        self.evaluate_personalized_model(grp)
        
        self.save_results()
        self.save_model()