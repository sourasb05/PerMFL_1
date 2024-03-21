#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverPerMFL import PerMFL
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.hierarchicalserveravg import hier_fedavg
from FLAlgorithms.servers.serverpFedMeoriginal import pFedMeoriginal
from FLAlgorithms.servers.serverL2GD import L2GD
from FLAlgorithms.servers.ditto_server import ditto_server
from FLAlgorithms.trainmodel.models import *
from FLAlgorithms.servers.serverpFedbayes import pFedBayes
from FLAlgorithms.servers.server_hqsgd import QSGD_server
#from utils.result_utils import average_result
import os
torch.manual_seed(0)


def main(dataset, algorithm, model_name, batch_size, beta, lamda, gamma, eta, alpha,
        num_glob_iters,local_iters, team_epochs, optimizer, numusers, times, gpu, K,
        personal_learning_rate, i, num_teams, tot_users, num_labels, analysis, group_division,
        weight_scale, rho_offset, zeta, p_teams):
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    
    while i < times:
        print("---------------Running time:------------", i)
    # Generate model
        if model_name == "mclr":
            if dataset == "Mnist":
                model = Mclr_Logistic().to(device)
            elif dataset == "FMnist":
                model = Mclr_Logistic(784, 10).to(device)
            elif dataset == "Emnist":
                model = Mclr_Logistic(input_dim=784, output_dim=62).to(device)
            elif dataset == "Cifar100":
                model = Mclr_Logistic(input_dim=3072, output_dim=100).to(device)
            else:
                model = Mclr_Logistic(60, 10).to(device)
        if model_name == "cnn":
            if dataset == "Mnist":
                model = Net().to(device)
            elif dataset == "Cifar10":
                model = CNNCifar(10).to(device)
            elif dataset == "FMnist":
                model = cnn_fmnist(10).to(device)
            elif dataset == "Emnist":
                model = cnn_emnist().to(device)
            elif dataset == "Cifar100":
                model = CNNCifar100().to(device)

        if model_name == "dnn":
            if dataset == "Mnist":
                model = DNN().to(device)
            elif dataset == "Synthetic":
                model = DNN(60, 20, 10).to(device)

        if model_name == "pbnn":
            if dataset == "Mnist":
                model = pBNN(784, 100, 10, device, weight_scale, rho_offset, zeta).to(device)
            else:
                model = pBNN(3072, 100, 10, device, weight_scale, rho_offset, zeta).to(device)

        if model_name in ("VGG11", "VGG13", "VGG16", "VGG19"):
            model = VGG(model_name).to(device)
            # print(model[0])

        # select algorithm
        # Single layer federated avg
        if algorithm == "FedAvg":
            server = FedAvg(device,
                            dataset,
                            algorithm,
                            model,
                            model_name,
                            batch_size,
                            alpha,
                            num_glob_iters,
                            local_iters,
                            optimizer,
                            numusers,
                            tot_users,
                            num_labels,
                            num_teams,
                            group_division,
                            i)
        # Proposed algorithm
        # Multi-layer personalized Federated learning
        
        if algorithm == "PerMFL":
            server = PerMFL(device,
                            dataset,
                            algorithm,
                            model,
                            batch_size,
                            alpha,
                            beta,
                            lamda,
                            gamma,
                            eta,
                            num_glob_iters,
                            local_iters,team_epochs,
                            optimizer,
                            numusers,
                            model_name,
                            i,
                            num_teams,
                            tot_users,
                            num_labels,
                            analysis,
                            group_division,
                            p_teams)
            
        #  Hierarchical Local SGD with Quantization 

        if algorithm == "Hier_local_qsgd" :
            server = QSGD_server(device,
                                 dataset,
                                 algorithm,
                                 model,
                                 batch_size,
                                 alpha,
                                 num_glob_iters,
                                 local_iters,
                                 team_epochs,
                                 optimizer,
                                 numusers,
                                 model_name,
                                 i)

        # Personalized- FedAvg

        if algorithm == "PerAvg":
            server = PerAvg(device,
                            dataset,
                            algorithm,
                            model,
                            model_name,
                            batch_size,
                            alpha,
                            beta,
                            lamda,
                            num_glob_iters,
                            local_iters,
                            optimizer,
                            numusers,
                            tot_users,
                            num_labels,
                            num_teams,
                            group_division,
                            i)

        # pFedMe original

        if algorithm == "pFedMe_original":
            server = pFedMeoriginal(device,
                                    dataset,
                                    algorithm,
                                    model,
                                    model_name,
                                    batch_size, 
                                    alpha,
                                    beta,
                                    lamda,
                                    num_glob_iters,
                                    local_iters,
                                    optimizer,
                                    numusers,
                                    tot_users,
                                    num_labels,
                                    num_teams,
                                    group_division,
                                    K,
                                    personal_learning_rate,
                                    i)

        # hierarchical Fed-Avg

        if algorithm == "hierarchical-FedAvg":
            server = hier_fedavg(device,
                                dataset,
                                algorithm,
                                model,
                                model_name,
                                batch_size,
                                alpha,
                                num_glob_iters,
                                local_iters,
                                team_epochs,
                                optimizer,
                                numusers,
                                tot_users,
                                num_labels,
                                num_teams,
                                group_division,
                                i)

        # Async-L2GD
 
        if algorithm == "AL2GD":
            server = L2GD(device, 
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
                        num_teams,
                        numusers,
                        i)
        
        # pFedBayes


        if algorithm == "pFedBayes":
            server = pFedBayes(dataset,
                               algorithm,
                               model,
                               model_name,
                               batch_size,
                               alpha,
                               beta,
                               lamda,
                               num_glob_iters,
                               local_iters,
                               optimizer,
                               tot_users,
                               times,
                               device,
                               num_labels,
                               num_teams,
                               group_division,
                               personal_learning_rate,
                               output_dim=10,
                               post_fix_str=''
                               )
        
        # ditto

        if algorithm == "ditto":
            server = ditto_server(device,
                                dataset,
                                algorithm, 
                                model, 
                                model_name, 
                                batch_size, 
                                alpha, 
                                lamda,
                                num_glob_iters,
                                local_iters,
                                tot_users, 
                                numusers, 
                                num_labels, 
                                num_teams, 
                                group_division,
                                i)
            

        
        server.train()
        
        i+=1
    # if algorithm != "PerMFL":
    #    path = "./results/"+ algorithm + "/" + dataset + "/" + model_name + "/" 
    #else:
    #    path = "./results/"+ algorithm + "/" + dataset + "/" + model_name + "/" + analysis + "/"    
    # average_result(path, algorithm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Synthetic", choices=["Mnist",
                                                                        "Synthetic",
                                                                        "Cifar10", 
                                                                        "FMnist",  
                                                                        "Emnist",
                                                                        "Cifar100",
                                                                        "Movielens"])
    parser.add_argument("--model_name", type=str, default="dnn", choices=["dnn",
                                                                            "mclr",
                                                                            "cnn",
                                                                            "pbnn",
                                                                            "VGG11",
                                                                            "VGG13",
                                                                            "VGG16",
                                                                            "VGG19"])
    parser.add_argument("--batch_size", type=int, default=124)
    parser.add_argument("--beta", type=float, default=0.99,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=float, default=5.0, 
                        help="Regularization term lambda")
    parser.add_argument("--gamma", type=float, default=5.0, 
                        help="regularization term gamma")
    parser.add_argument("--alpha", type=float, default=0.01, 
                        help="learning rate")
    parser.add_argument("--eta", type=float, default=0.03, 
                        help="regularization term eta")
    
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--num_team_iters", type=int, default=10)
    parser.add_argument("--local_iters", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="hierarchical-FedAvg",
                        choices=["PerMFL",
                                  "PerAvg", 
                                  "FedAvg", 
                                  "pFedMe_original", 
                                  "hierarchical-FedAvg", 
                                  "AL2GD", 
                                  "pFedBayes", 
                                  "ditto",
                                  "Hier-Local-QSGD"])
    
    
    parser.add_argument("--K", type=int, default=5, 
                        help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.005,
                        help="Personalized learning rate to calculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, 
                        help="running time")
    parser.add_argument("--exp_start", type=int, default=0,
                        help="experiment start no")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--tot_users", type=int, default=40, 
                        help="Total number of users")
    parser.add_argument("--selected_users", type=int, default=10, 
                        help="selected user per round of training")
    parser.add_argument("--numusers", type=int, default=10, 
                        help="Number of Users per round")
    parser.add_argument("--num_teams", type=int, default=4,
                        help="Number of teams")
    parser.add_argument("--p_teams", type=int, default=4,
                        help="number of team selected per global round")
    parser.add_argument("--num_labels",type=int, default=2,
                        help="number of labels per user" )
    parser.add_argument("--analysis", type = str, default="perf",
                        help="What kind of analysis do you want (beta, gamma, lambda, eta, team_iters, team_number, poor_cluster)")
    
    parser.add_argument("--group_division", type = int, default=1, help=" 0 : sequential division , 1 : random division , 2 : only one group")
   
    parser.add_argument("--weight_scale", type=float, default=0.1)
    parser.add_argument("--rho_offset", type=int, default=-3)
    parser.add_argument("--zeta", type=int, default=10)

    args = parser.parse_args()

    print("=" * 90)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Total users: {}".format(args.tot_users))
    print("Number of labels per user: {}".format(args.num_labels))
    print("alpha       : {}".format(args.alpha))
    print("beta        : {}".format(args.beta))
    print("gamma       : {}".format(args.gamma))
    print("lamda       : {}".format(args.lamda))
    print("number of teams : {}".format(args.num_teams))
    # print("eta         : {}".format(args.eta))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of team rounds          : {}".format(args.num_team_iters))
    print("Number of local rounds       : {}".format(args.local_iters))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model_name))
    print("=" * 80)

    if args.algorithm == "pFedMe_original":
        print("K : {}".format(args.K))
        print("personal learning rate : {}".format(args.personal_learning_rate))
    
    if args.algorithm == "pFedBayes":
        print("weight scale : {}".format(args.weight_scale))
        print("rho_offset : {}".format(args.rho_offset))
        print("zeta : {}".format(args.zeta))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model_name=args.model_name,
        batch_size=args.batch_size,
        beta=args.beta,
        lamda=args.lamda,
        gamma=args.gamma,
        eta=args.eta,
        alpha=args.alpha,
        num_glob_iters=args.num_global_iters,
        team_epochs=args.num_team_iters,
        local_iters=args.local_iters,
        optimizer=args.optimizer,
        numusers=args.numusers,
        times=args.times,
        gpu=args.gpu,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        i=args.exp_start,
        num_teams=args.num_teams,
        tot_users=args.tot_users,
        num_labels=args.num_labels,
        analysis=args.analysis,
        group_division=args.group_division,
        weight_scale=args.weight_scale,
        rho_offset=args.rho_offset,
        zeta=args.zeta,
        p_teams=args.p_teams
    )
