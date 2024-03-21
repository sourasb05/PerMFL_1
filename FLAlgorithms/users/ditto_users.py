from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import DataLoader 
from FLAlgorithms.optimizers.fedoptimizer import Ditto_Optimizer
import copy
import torch


class DittoUser():
    def __init__(self,
                 device,
                 user_id,
                 trainset,
                 testset,
                 model,
                 batch_size,
                 learning_rate,
                 ditto_lambda,
                 num_local_iters):
        
        self.device = device
        self.user_id = user_id
        self.trainset = trainset
        self.testset = testset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ditto_lambda = ditto_lambda
        self.local_iters = num_local_iters

        self.train_samples = len(trainset)
        self.test_samples = len(testset)

        self.trainloader = DataLoader(trainset, self.batch_size)
        self.testloader = DataLoader(testset, self.batch_size)
        self.trainloaderfull = DataLoader(trainset, self.train_samples)
        self.testloaderfull = DataLoader(testset, self.test_samples)
        
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        
        self.global_model = copy.deepcopy(model)
        self.local_model = copy.deepcopy(model)
        self.v_k = copy.deepcopy(model)
        self.delta_tk = copy.deepcopy(model)

        self.per_model_param = copy.deepcopy(list(self.local_model.parameters()))
        self.optimizer = Ditto_Optimizer(self.local_model.parameters(), eta=self.learning_rate, ditto_lambda=self.ditto_lambda)
        self.criterion = SGD(self.local_model.parameters(), lr=self.learning_rate)
        self.loss = nn.CrossEntropyLoss()

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


    def get_parameters(self, global_model):
        for local_param, global_param in zip(self.local_model.parameters(), global_model.parameters()):
           local_param.data = global_param.data 

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def train(self):
        for iter in range(self.local_iters):
            self.local_model.train()
            X, y = self.get_next_train_batch()
            self.criterion.zero_grad()
            output = self.local_model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.criterion.step()

    def ditto_train(self, global_model):
        for iter in range(self.local_iters):
            self.v_k.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.v_k(X)
            loss = self.loss(output, y)
            loss.backward()
            self.per_model_param, _ = self.optimizer.step(self.v_k.parameters(), global_model.parameters())
            
            for new_param, per_param in zip(self.per_model_param, self.v_k.parameters()):
                per_param.data = new_param.data

    def calc_delta_tk(self, global_model):
        for d_tk, local_param, global_param in zip(self.delta_tk.parameters(), self.local_model.parameters(), global_model.parameters()):
            d_tk.data = local_param.data - global_param.data 

    def update_parameters(self, new_params):
        for param, new_param in zip(self.global_model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_personalized_model(self):
        self.v_k.eval()
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.v_k(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        
        return test_acc, loss, y.shape[0]
    
    def train_error_and_loss_personalized_model(self):
        self.v_k.eval()
        train_acc = 0
        loss = 0
        # self.update_parameters(self.personalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.v_k(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        # self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples
    
    def test(self, eval_glob_param):
        self.global_model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(eval_glob_param)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.global_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id, " + , Test Accuracy:", test_acc / int(y.shape[0]) )
            # print(self.id, " + , Test Loss:", loss)
        return test_acc, loss, y.shape[0]
    

    def train_error_and_loss(self, eval_glob_param):
        self.global_model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(eval_glob_param)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.global_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        return train_acc, loss, self.train_samples

