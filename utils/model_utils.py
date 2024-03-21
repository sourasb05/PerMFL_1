import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import random
import time
import pandas as pd
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3


def suffle_data(data):
    data_x = data['x']
    data_y = data['y']
    # randomly shuffle data
    np.random.seed(42)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x) // batch_size + 1
    if len(data_x) > batch_size:
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx * batch_size
        if sample_index + batch_size > len(data_x):
            return data_x[sample_index:], data_y[sample_index:]
        else:
            return data_x[sample_index: sample_index + batch_size], data_y[sample_index: sample_index + batch_size]
    else:
        return data_x, data_y


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_cifa_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data

    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    # NUM_USERS = 20  # should be muitiple of 10
    # NUM_LABELS = 2
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)
    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label == i

        cifa_data.append(cifa_data_image[idx])

      
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)  # number of classes
   
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  
            l = (user + j) % 10
            X[user] += cifa_data[l][idx[l]:idx[l] + 10].tolist()
            y[user] += (l * np.ones(10)).tolist()
            idx[l] += 10
    
    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v) - NUM_USERS]] for v in cifa_data]) * \
            props / np.sum(props, (1, 2), keepdims=True)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user // int(NUM_USERS / 10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples) + numran1  # + 200
            if (NUM_USERS <= 20):
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l] + num_samples].tolist()
                y[user] += (l * np.ones(num_samples)).tolist()
                idx[l] += num_samples

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    group = [[] for _ in range(NUM_GROUPS)]
    cl_per_grp = NUM_USERS // NUM_GROUPS
    grp_idx = 0

    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        print(len(combined))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len

        test_data['users'].append(uname)
        test_data["user_data"][uname] = {'x': X[i][:test_len], 'y': y[i][:test_len]}
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] = {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

        # sequential group division type
        if group_division == 0:
            group[grp_idx].append(i)
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1

        # random group division type
    if group_division == 1:
        user_list = list(range(0,NUM_USERS))
        user_list = random.shuffle(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1

    if group_division == 2:
        user_list = list(range(0,NUM_USERS))
        user_list = random.shuffle(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
    
    
    return train_data['users'], group, train_data['user_data'], test_data['user_data']


def read_Mnist_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader, 0):
        testset.data, testset.targets = train_data

    random.seed(5)
    np.random.seed(9)
    # NUM_USERS = 20  # should be muitiple of 10
    # NUM_LABELS = 2
    # Setup directory for train/test data
    train_path = './data/train/mnist_train_100.json'
    test_path = './data/test/mnist_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)
    # print(cifa_data_image)
    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label == i
        # print(idx)
        cifa_data.append(cifa_data_image[idx])
        # print("len(cifa_data[", i, "] :", len(cifa_data[i]))

    # print("\n Numb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign samples to randomly to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    # print(X)
    # print(y)
    # input("Press")
    idx = np.zeros(10, dtype=np.int64)  # number of classes
    # print(idx)
    # NUM_USERS = 20
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 2 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            # print("L:", l)
            # l_max = random.randint(1500, 3000)
            X[user] += cifa_data[l][idx[l]:idx[l] + 10].tolist()
            y[user] += (l * np.ones(10)).tolist()
            idx[l] += 10
            # print("len X[", user, "] =", len(X[user]))
            # print("len y[", user, "] =", len(y[user]))
            # print("idx[", l, "] =", idx[l])
    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v) - NUM_USERS]] for v in cifa_data]) * \
            props / np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    # props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    # idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user // int(NUM_USERS / 10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples) + numran1  # + 200
            if (NUM_USERS <= 20):
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l] + num_samples].tolist()
                y[user] += (l * np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j,
                #      "len data", len(X[user]), num_samples)

    # print("IDX2:", idx)  # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    group = [[] for _ in range(NUM_GROUPS)]
    cl_per_grp = NUM_USERS // NUM_GROUPS
    grp_idx = 0
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        print(len(combined))
        # print(combined[0])
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len

        # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\

        test_data['users'].append(uname)
        test_data["user_data"][uname] = {'x': X[i][:test_len], 'y': y[i][:test_len]}
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] = {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

        # sequential group division type
        if group_division == 0:
            group[grp_idx].append(i)
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1

        # random group division type
    user_list = []
    if group_division == 1:
        
        for i in range(NUM_USERS):
            user_list.append(i)
        current_time = time.time()
        random.seed(current_time)
        print(current_time)
        print(user_list)

   
        
        random.shuffle(user_list)
        print(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1
    
    if group_division == 2:
        user_list = list(range(0,NUM_USERS))
        user_list = random.shuffle(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
    
    return train_data['users'], group, train_data['user_data'], test_data['user_data']



def read_FMnist_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader, 0):
        testset.data, testset.targets = train_data

    random.seed(5)
    np.random.seed(9)
    train_path = './data/train/fmnist_train_100.json'
    test_path = './data/test/fmnist_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)
    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label == i
        cifa_data.append(cifa_data_image[idx])
    
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign samples to randomly to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)  # number of classes
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 2 labels for each users
            l = (user + j) % 10
            X[user] += cifa_data[l][idx[l]:idx[l] + 10].tolist()
            y[user] += (l * np.ones(10)).tolist()
            idx[l] += 10
    
    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v) - NUM_USERS]] for v in cifa_data]) * \
            props / np.sum(props, (1, 2), keepdims=True)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user // int(NUM_USERS / 10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples) + numran1  # + 200
            if (NUM_USERS <= 20):
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l] + num_samples].tolist()
                y[user] += (l * np.ones(num_samples)).tolist()
                idx[l] += num_samples
    
    
    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    group = [[] for _ in range(NUM_GROUPS)]
    cl_per_grp = NUM_USERS // NUM_GROUPS
    grp_idx = 0

    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len


        test_data['users'].append(uname)
        test_data["user_data"][uname] = {'x': X[i][:test_len], 'y': y[i][:test_len]}
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] = {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
        # sequential group division type
        if group_division == 0:
            group[grp_idx].append(i)
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1

        # random group division type
    user_list = []
    if group_division == 1:
        
        for i in range(NUM_USERS):
            user_list.append(i)
        print(user_list)
        current_time = time.time()
        random.seed(current_time)
        print(current_time)
        random.shuffle(user_list)
        print(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1

                
    if group_division == 2:
        user_list = list(range(0,NUM_USERS))
        user_list = random.shuffle(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
    
    return train_data['users'], group, train_data['user_data'], test_data['user_data']


###############################################

"""
Read MovieLens dataset

"""
#################################################

def read_MovieLens_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division):
    DATA_DIR = './data/ml-latest-small/'
    dfm = pd.read_csv(DATA_DIR+'movies.csv')
    df = pd.read_csv(DATA_DIR+'ratings.csv')
    df = df.merge(dfm, on='movieId', how='left')
    df = df.sort_values(['userId', 'timestamp'], ascending=[True, True]).reset_index(drop=True)




##########################################################################
""""
Read EMNIST data

"""
##########################################################################
def read_EMnist_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division):
    transform = transforms.Compose(
        [transforms.ToTensor(), # convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))
        ])  # normalize the data


    # Download the EMNIST dataset and apply the transformation

    """
    We can split 
    digits that contains 10 classes,
    byClass contains 62 classes,
    byMerge contains 47 classes,
    Balanced contains 47 classes,
    Letters contains 26 classes,
    """
    trainset = torchvision.datasets.EMNIST(root='./data', train=True, split='byclass', download=True, transform=transform)
    testset = torchvision.datasets.EMNIST(root='./data', train=False,split='byclass', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0
    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    
    train_path = './data/train/emnist_train.json'
    test_path = './data/test/emnist_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    emnist_data_image = []
    emnist_data_label = []
    
    emnist_data_image.extend(trainset.data.cpu().detach().numpy()) 
    emnist_data_image.extend(testset.data.cpu().detach().numpy())
    emnist_data_label.extend(trainset.targets.cpu().detach().numpy())
    emnist_data_label.extend(testset.targets.cpu().detach().numpy())
    emnist_data_image = np.array(emnist_data_image)
    emnist_data_label = np.array(emnist_data_label)

    emnist_data = []
    for i in trange(62):
        idx = emnist_data_label==i
        emnist_data.append(emnist_data_image[idx])


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 10 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(62, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  
            #l = (2*user+j)%10
            l = (user + j) % 62
            # print("L:", l)
            X[user] += emnist_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10
            print(idx)

    # print("IDX1:", idx)  # counting samples for each labels
    # input("press at 587")
    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (62, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in emnist_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 62
            
            # num_samples = int(props[l, user//int(NUM_USERS/62), j])
            # print(num_samples)
            # print("j = ",j)
            # input("press")
            numran1 = random.randint(100,3000)
            # num_samples = (num_samples)  + numran1 #+ 200
            num_samples = numran1
            # print(num_samples)
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(emnist_data[l]):
                print(idx[l] + num_samples)
                print(len(emnist_data[l]))
                X[user] += emnist_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    group = [[] for _ in range(NUM_GROUPS)]
    cl_per_grp = NUM_USERS // NUM_GROUPS
    grp_idx = 0
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        # random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        

        # sequential group division type
        if group_division == 0:
            group[grp_idx].append(i)
            if i == 10 or i == 36 :
                grp_idx+=1

        # random group division type
    # random group division type
    user_list = []
    if group_division == 1:
        for i in range(NUM_USERS):
            user_list.append(i)
        
        current_time = time.time()
        random.seed(current_time)
        print(current_time)
        
        print(user_list)
        random.shuffle(user_list)
        print(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1
           
        # unequal group division
    if group_division == 2:
        random.seed(42)
        user_list = list(range(0,NUM_USERS))
        user_list = random.shuffle(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])

    
    
    return train_data['users'], group, train_data['user_data'], test_data['user_data']

##########################################################################
"""
Read Celeba data

"""
##########################################################################

def read_Celeba_data():
    # Define transformations to be applied to the input images
    transform = transforms.Compose([ 
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    testset = torchvision.datasets.CelebA(root='./data', split='test', download=True, transform=transform)
    trainloader = torch.utils.data.CelebA(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.CelebA(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0
    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 1000 # should be muitiple of 10
    NUM_LABELS = 10
    # Setup directory for train/test data
    train_path = './data/train/celeba_train.json'
    test_path = './data/test/celeba_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    celeba_data_image = []
    celeba_data_label = []
    
    celeba_data_image.extend(trainset.data.cpu().detach().numpy()) 
    celeba_data_image.extend(testset.data.cpu().detach().numpy())
    celeba_data_label.extend(trainset.targets.cpu().detach().numpy())
    celeba_data_label.extend(testset.targets.cpu().detach().numpy())
    celeba_data_image = np.array(celeba_data_image)
    celeba_data_label = np.array(celeba_data_label)

    celeba_data = []
    for i in trange(10):
        idx = celeba_data_label==i
        celeba_data.append(celeba_data_image[idx])


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            # print("L:", l)
            X[user] += celeba_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in celeba_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(celeba_data[l]):
                X[user] += celeba_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    return train_data['users'], train_data['user_data'], test_data['user_data']

#########################################################################
"""
Synthetic dataset creation started.......

"""
#########################################################################



def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(NUM_USER, alpha, beta, iid):
    dimension = 60
    NUM_CLASS = 10
    samples_per_user = (np.random.lognormal(4, 2, NUM_USER).astype(int) + 50) * 5
    # print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        # print(mean_x[i])

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1, NUM_CLASS)

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1, NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        # print("{}-th users has {} exampls".format(i, len(y_split[i])))

    return X_split, y_split


def read_synthetic_data(NUM_USERS, NUM_GROUPS, group_division):
    np.random.seed(0)
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    train_path = "data/train/mytrain.json"
    test_path = "data/test/mytest.json"
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # NUM_USERS = 50
    X, y = generate_synthetic(NUM_USERS, alpha=0.5, beta=0.5, iid=0)  # synthetic (0.5, 0.5)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    group = [[] for _ in range(NUM_GROUPS)]
    cl_per_grp = NUM_USERS // NUM_GROUPS
    grp_idx = 0
    for i in trange(NUM_USERS, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

        # sequential group division type
        if group_division == 0:
            group[grp_idx].append(i)
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1

    # random group division type
    user_list = []
    if group_division == 1:

        
        for i in range(NUM_USERS):
            user_list.append(i)
        
        print(user_list)
        current_time = time.time()
        random.seed(current_time)
        print(current_time)
        random.shuffle(user_list)
        print(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1



    if group_division == 2:
        user_list = list(range(0,NUM_USERS))
        user_list = random.shuffle(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            
         

    return train_data['users'], group, train_data['user_data'], test_data['user_data']



def read_cifar100_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0

    """

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    # NUM_USERS = 100 # should be muitiple of 10
    # NUM_LABELS = 10
    # Setup directory for train/test data
    train_path = './data/train/cifar100_train.json'
    test_path = './data/test/cifar100_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifar100_data_image = []
    cifar100_data_label = []
    
    cifar100_data_image.extend(trainset.data.cpu().detach().numpy()) 
    cifar100_data_image.extend(testset.data.cpu().detach().numpy())
    cifar100_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifar100_data_label.extend(testset.targets.cpu().detach().numpy())
    cifar100_data_image = np.array(cifar100_data_image)
    cifar100_data_label = np.array(cifar100_data_label)

    cifar100_data = []
    for i in trange(100):
        idx = cifar100_data_label==i
        cifar100_data.append(cifar100_data_image[idx])

    #print(len(cifar100_data))
    # input("press")


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(100, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 
            l = (user + j) % 100
            X[user] += cifar100_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()

            
            idx[l] += 10

    # Assign remaining sample by power law
    user = 0
    # props = np.random.lognormal(-2, 2, (100, NUM_USERS, NUM_LABELS))  
    
    # print("sm_props :",np.sum(props, (1, 2), keepdims=True))
    # input("press 304")
    # print(np.array([[[len(v)-NUM_USERS]] for v in cifar100_data]))
    # input()
    # props = np.array([[[len(v)-NUM_USERS]] for v in cifar100_data]) * \
    #     props/np.sum(props, (1, 2), keepdims=True)
    # print("Props :",props)
    # input("press at 168")
    
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 100
            # num_samples = int(props[l, user//int(NUM_USERS/100), j])
            # print("depth :",l)
            # print("row :",user//int(NUM_USERS/100))
            # print("Column :",j)
            # print("num_samples",num_samples)
            # input("press")
            numran1 = random.randint(300, 600)
            # num_samples = (num_samples) +  numran1 #+ 200
            # print(" num_samples", num_samples)
            num_samples = numran1


            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            # print("len(cfar100_data[",l,"] :", len(cifar100_data[l]))
            if idx[l] + num_samples < len(cifar100_data[l]):
                X[user] += cifar100_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples

                # print("y[",user,"] :",y[user])
                # print("idx[",l,"] :",idx[l])

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    group = [[] for _ in range(NUM_GROUPS)]
    cl_per_grp = NUM_USERS // NUM_GROUPS
    grp_idx = 0
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    # sequential group division type
        if group_division == 0:
            group[grp_idx].append(i)
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1

    # random group division type
    user_list = []
    if group_division == 1:

        
        for i in range(NUM_USERS):
            user_list.append(i)
        
        print(user_list)
        current_time = time.time()
        random.seed(current_time)
        print(current_time)
        random.shuffle(user_list)
        print(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            if i != 0 and (i+1) % cl_per_grp == 0 :
                grp_idx+=1



    if group_division == 2:
        user_list = list(range(0,NUM_USERS))
        user_list = random.shuffle(user_list)
        for i in range(NUM_USERS):
            group[grp_idx].append(user_list[i])
            
         

    return train_data['users'], group, train_data['user_data'], test_data['user_data']




def read_data(dataset, num_users, num_labels, num_groups, group_division):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    if dataset == "Cifar10":
        print("at read data")
        clients, groups, train_data, test_data = read_cifa_data(num_users, num_labels, num_groups, group_division)
        print("clients", clients)
        print("groups", groups)
        # return clients, groups, train_data, test_data
    elif dataset == "FMnist":
        print("at read data")
        clients, groups, train_data, test_data = read_FMnist_data(num_users, num_labels, num_groups, group_division)
        print("clients", clients)
        print("groups", groups)
        # return clients, groups, train_data, test_data
    elif dataset == "Mnist":
        print("reading from Mnist data")
        clients, groups, train_data, test_data = read_Mnist_data(num_users, num_labels, num_groups, group_division)
        print("clients", clients)
        print("groups", groups)
        # return clients, groups, train_data, test_data
    
    elif dataset == "Emnist":
        print("reading from Emnist data")
        clients, groups, train_data, test_data = read_EMnist_data(num_users, num_labels, num_groups, group_division)
        print("clients", clients)
        print("groups", groups)
        # return clients, groups, train_data, test_data

    elif dataset == "Cifar100":
        print("reading from Cifar100 data")
        clients, groups, train_data, test_data = read_cifar100_data(num_users, num_labels, num_groups, group_division)
        print("clients", clients)
        print("groups", groups)
        # return clients, groups, train_data, test_data

    else:
        print("reading from synthetic data")
        clients, groups, train_data, test_data = read_synthetic_data(num_users, num_groups, group_division)
        print("clients", clients)
        print("groups", groups)
        # return clients, groups, train_data, test_data

   
    return clients, groups, train_data, test_data













def read_user_data(index, data, dataset):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if (dataset == "Mnist"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif (dataset == "Cifar10"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    elif (dataset == "FMnist"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(
            torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(
            torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    elif(dataset == "EMNIST"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(
            torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(
            torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data
