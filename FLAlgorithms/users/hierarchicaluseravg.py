import torch
import torch.nn as nn
import copy
import os
from torch.utils.data import DataLoader


# Implementation for pFedMe clients

class hier_fedavg_user():
    def __init__(self,
                device,
                numeric_id,
                train_data,
                test_data,
                model,
                model_name,
                batch_size,
                alpha,
                local_epochs,
                dataset):
      
        if (model_name == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
            print("model_name :", model_name)
            # input("press :")
        elif model_name == "cnn" and dataset=="FMnist":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
       
        self.alpha = alpha
        self.device = device
        self.local_model = copy.deepcopy(model)
        self.id = numeric_id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.alpha = alpha
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader = DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.alpha)

    def set_parameters(self, model):
        for local_param, param in zip(self.local_model.parameters(), model.parameters()):
            local_param.data = param.data.clone()
       
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

    
    def train(self):

        for epoch in range(0, self.local_epochs):  # local update
            # print(" at training :: Epoch [", epoch, "]")
            self.local_model.train()
            X, y = self.get_next_train_batch()
            # print("X :", X, "y :", y)

            # K = 30 # K is number of personalized steps
            #  for i in range(self.K):
            self.optimizer.zero_grad()
            output = self.local_model(X)
            loss = self.loss(output, y)
            # print(loss)
            loss.backward()
            self.optimizer.step()
    

    def test(self, global_model):
        self.local_model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return test_acc, loss, y.shape[0]

    def train_error_and_loss(self, global_model):
        self.local_model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            
        return train_acc, loss, y.shape[0]