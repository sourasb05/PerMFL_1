import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

#################################
##### Logistic regression #######
#################################

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 62):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 62):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

########################################
########### Neural Network #############
########################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        # x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        # x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class cnn_fmnist(nn.Module):
  def __init__(self, args):
    super(cnn_fmnist, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.maxpool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.maxpool2 = nn.MaxPool2d(2, 2)
    self.flatten = nn.Flatten()
    self.dense1 = nn.Linear(3136, 128)
    self.dense2 = nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.maxpool1(x)
    x = F.relu(self.conv2(x))
    x = self.maxpool2(x)
    x = self.flatten(x)
    x = F.relu(self.dense1(x))
    x = self.dense2(x)
    return x


class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.dropout(.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    




class cnn_emnist(nn.Module):
    def __init__(self):
        super(cnn_emnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        # x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        # x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=.5)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],

                            ['conv3.weight', 'conv3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]
                            
    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.pool(F.relu(self.conv1(x)))  # 16*16*16
        x = self.pool(F.relu(self.conv2(x)))  # 8*8*32
        x = self.pool(F.relu(self.conv3(x)))  # 4*4*64
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # return x
        return F.log_softmax(x, dim=1)
    

    # Define the CNN architecture
class CNNCifar100(nn.Module):
    def __init__(self, num_classes=100):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

#########################################################
############## Deep Neural networks #####################
#########################################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



##########################################################
####################### pFedBayes ########################
##########################################################

class pBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cpu'),
                 weight_scale=0.1, rho_offset=-3, zeta=10):
        super(pBNN, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = 1
        self.mean_prior = 10
        self.sigma_prior = 5
        self.layer_param_shapes = self.get_layer_param_shapes()
        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = torch.tensor([1.] * len(self.layer_param_shapes), device=self.device)

        for shape in self.layer_param_shapes:
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)

    def get_layer_param_shapes(self):
        layer_param_shapes = []
        for i in range(self.num_layers + 1):
            if i == 0:
                W_shape = (self.input_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            elif i == self.num_layers:
                W_shape = (self.hidden_dim, self.output_dim)
                b_shape = (self.output_dim,)
            else:
                W_shape = (self.hidden_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes

    def transform_rhos(self, rhos):
        return [F.softplus(rho) for rho in rhos]

    def transform_gaussian_samples(self, mus, rhos, epsilons):
        # compute softplus for variance
        self.sigmas = self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)): samples.append(mus[j] + self.sigmas[j] * epsilons[j])
        return samples

    def sample_epsilons(self, param_shapes):
        epsilons = [torch.normal(mean=torch.zeros(shape), std=0.001*torch.ones(shape)).to(self.device) for shape in
                    param_shapes]
        return epsilons

    def net(self, X, layer_params):
        layer_input = X
        for i in range(len(layer_params) // 2 - 1):
            h_linear = torch.mm(layer_input, layer_params[2 * i]) + layer_params[2 * i + 1]
            layer_input = F.relu(h_linear)

        output = torch.mm(layer_input, layer_params[-2]) + layer_params[-1]
        return output

    def log_softmax_likelihood(self, yhat_linear, y):
        return torch.nansum(y * F.log_softmax(yhat_linear), dim=0)

    def combined_loss_personal(self, output, label_one_hot, params, mus, sigmas, mus_local, sigmas_local, num_batches):

        # Calculate data likelihood
        log_likelihood_sum = torch.sum(self.log_softmax_likelihood(output, label_one_hot))
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i], sigmas[i]),
                            Normal(mus_local[i].detach(), sigmas_local[i].detach())))  for i in range(len(params))])

        return 1.0 / num_batches * (self.zeta * KL_q_w) - log_likelihood_sum

    def combined_loss_local(self, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i].detach(), sigmas[i].detach()),
                        Normal(mus_local[i], sigmas_local[i]))) for i in range(len(params))])
        return 1.0 / num_batches * (self.zeta * KL_q_w)
