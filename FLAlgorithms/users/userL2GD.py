import torch.nn as nn
from FLAlgorithms.optimizers.fedoptimizer import L2GDopt
import copy
import os
import torch
from torch.utils.data import DataLoader




# from tqdm import trange


# Implementation for pFedMe clients

class UserL2GD():
    def __init__(self, 
                device, 
                id,
                train_data,
                test_data,
                model,
                model_name,
                batch_size, 
                alpha,
                gamma,
                tau,
                p_0,
                p_j,
                eta,
                local_iters):
        
        if (model_name == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.device = device
        self.id = id
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.p_0 = p_0
        self.p_j = p_j
        self.eta = eta
        self.local_epochs = local_iters
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader = DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        self.local_model = copy.deepcopy(model)
        self.theta_i_t1 = copy.deepcopy(list(self.local_model.parameters()))
        self.optimizer = L2GDopt(self.local_model.parameters(), lr=self.eta, p_0=self.p_0, p_j=self.p_j)


    def set_parameters(self, model):
        for local_param, param in zip(self.local_model.parameters(), model.parameters()):
            local_param.data = param.data.clone()
        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.local_model.parameters():
            param.detach()  ## to remove the require_grad and only fetch the tensor
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()

    
    
    def test(self, global_model):
        local_model_params = copy.deepcopy(list(self.local_model.parameters()))
        self.local_model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        self.update_parameters(local_model_params)
        return test_acc, loss, y.shape[0]
    
    def train_error_and_loss(self, global_model):
        local_model_params = copy.deepcopy(list(self.local_model.parameters()))
        self.local_model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(local_model_params)
        return train_acc, loss, self.train_samples

    
    def train_error_and_loss_personalized_model(self):
        self.local_model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.local_model.parameters())
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model.parameters())
        return train_acc, loss, self.train_samples

    def test_personalized_model(self):
        self.local_model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.local_model.parameters())
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        
        self.update_parameters(self.local_model.parameters())
        return test_acc, loss, y.shape[0]

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
            # print("X :", len(X), "y :", len(y))

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
            # print("In exception X :", len(X), "y :", len(y))
        return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.local_model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.local_model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))

    def compute_theta_1(self, global_model, cluster_model):
        for l_param, glob_param, cluster_param in zip(self.local_model.parameters(), 
                                                    global_model.parameters(), 
                                                    cluster_model.parameters()):
                l_param.data = (1 - (self.eta* self.gamma*(self.alpha + self.tau*(1-self.alpha))/self.p_0))*l_param.data 
                + ((self.eta * self.gamma)/self.p_0) *(self.alpha * glob_param.data + self.tau * (1- self.alpha)*cluster_param.data) 
        

    def compute_theta_2(self, cluster_model):

        for l_param, cluster_param in zip(self.local_model.parameters(), 
                                          cluster_model.parameters()):
            l_param.data = (1 - (self.eta*self.gamma*(1-self.tau)*(1-self.alpha))/((1-self.p_0)*self.p_j))*l_param.data
            + (self.eta*self.gamma*(1-self.tau)*(1-self.alpha)/(1-self.p_0)*self.p_j)*cluster_param.data



    def local_train(self, iters ):
        for iter in range(iters):  # local update
            self.local_model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.local_model(X)
            loss = self.loss(output, y)
            loss.backward()

            self.theta_i_t1, _ = self.optimizer.step(self.local_model.parameters(), self.eta, self.p_0, self.p_j)

            for new_param, localweight in zip(self.theta_i_t1, self.local_model.parameters()):
                localweight.data = new_param.data
        
        
        