import torch.nn as nn
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
from tqdm import trange
import copy

# Implementation for pFedMe clients

class UserPerMFL(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, model_name, 
                batch_size, alpha, beta, lamda, local_epochs, dataset):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, alpha, beta, lamda,
                         local_epochs)
        # print("model[1] :", model[1])
        # input("press :")
        if (model_name == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
            # print("model name :", model_name)
        elif model_name == "cnn" and dataset in ["FMnist", "Cifar100"]:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
            # self.loss = nn.CrossEntropyLoss()

        # self.K = K
        self.alpha = alpha
        self.optimizer = pFedMeOptimizer(self.model.parameters(), alpha=self.alpha, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs, team_model):
        team_model_list = copy.deepcopy(list(team_model))
        for epoch in range(0, epochs):  # local update
            # print(" at training :: Epoch [", epoch, "]")
            self.model.train()
            X, y = self.get_next_train_batch()
            # print("X :", X, "y :", y)

            # K = 30 # K is number of personalized steps
            #  for i in range(self.K):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            # print(loss)
            loss.backward()

            # personalized_model_bar is the weights that are generated after pFedMe operations in the client (lower)
            # level

            self.personalized_model_bar, _ = self.optimizer.step(team_model_list)

            # update local weight after finding aproximate theta. Copy the personalized_model_bar to the local model
            # of each user

            for new_param, localweight in zip(self.personalized_model_bar, self.local_model):
                # localweight.data = localweight.data - self.lamda * self.alpha * (
                #             localweight.data - new_param.data)
                localweight.data = new_param.data
        # update local model as local_weight_upated
        
        self.update_parameters(self.local_model)
