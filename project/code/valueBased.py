import torch
from torch import nn
import numpy as np
import random
from torch.nn import functional as F
from torchvision import models
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
}



'''
DQN
'''
class cnn(nn.Module):
    def __init__(self,inch, outdim):
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=inch, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2 )
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linerlayer = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),

        )
        self.outlayerA = nn.Linear(512,outdim)


    def forward(self,x):
        x = x/255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        x = self.linerlayer(x)
        x = self.outlayerA(x)
        return x


class dnn(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(dnn, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(inputdim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.outlayerA = nn.Linear(256, outputdim)
        self.outlayerV = nn.Linear(256, 1)


    def forward(self, x):
        x = self.layer(x)
        a = self.outlayerA(x)
        v = self.outlayerV(x)
        return v + (a - torch.mean(a))

class ReplayBuffer(object):

    def __init__(self, buffersize, batchsize):
        self.size = buffersize
        self.batchsize = batchsize
        self.buffer = np.zeros(self.size, dtype=object)
        self.pointer = 0

    def push(self,state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer[self.pointer] = transition
        self.pointer = (self.pointer+1) % self.size
    def get_batches(self):
        Batch = np.random.choice(self.buffer, self.batchsize,replace=False)
        return Batch

class ValueBasedModel(object):

    def __init__(self, buffer_size, batch_size, action_dim, modelName, lr, updatestep, inchannel=1, inputdim=128):
        self.batch_size = batch_size
        self.replayBuffer = ReplayBuffer(buffer_size, batch_size)
        self.epsilon_start = 1.0
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        self.epsi_decay = 1000000
        self.gamma = 0.99
        self.action_dim = action_dim
        self.inchannel = inchannel
        self.name = modelName
        self.updatestep = updatestep
        self.inputdim = inputdim

        self.eval_net, self.target_net = self.build_net()
        self.eval_net.to(device)
        self.target_net.to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        self.loss_fun = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr,eps = 1.5e-4)

    def build_net(self):
        if self.name == 'VideoPinball-ramNoFrameskip-v4':
            net1 = dnn(self.inputdim, self.action_dim)
            net2 = dnn(self.inputdim, self.action_dim)
            return net1, net2

        else:
            net1 = cnn(self.inchannel, self.action_dim)
            net2 = cnn(self.inchannel, self.action_dim)
            return net1, net2


    def get_action(self, obs, mode=True):
        '''
        :param obs: 已经处理好的形状
        :param mode: 是否探索
        :return: action
        '''
        q_val = self.target_net(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
        if mode:
            self.epsilon -= (self.epsilon_start-self.epsilon_end)/self.epsi_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)
            if random.random() < self.epsilon:
                return np.random.randint(self.action_dim)
            else:
                return np.argmax(q_val)
        else:
            if random.random() < 0.05:
                return np.random.randint(self.action_dim)
            else:
                return np.argmax(q_val)
    def train(self):

        sample = self.replayBuffer.get_batches()

        state_batch = torch.FloatTensor(np.array([_[0] for _ in sample])).to(device)
        action_batch = torch.LongTensor(np.array([_[1] for _ in sample])).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor(np.array([_[2] for _ in sample])).view(-1, 1).to(device)
        nextstate_batch = torch.FloatTensor(np.array([_[3] for _ in sample])).to(device)
        done_batch = torch.FloatTensor(np.array([_[4] for _ in sample])).view(-1, 1).to(device)

        Q_eval = self.eval_net(state_batch).gather(1, action_batch)
        Q_next = self.target_net(nextstate_batch).detach()
        #Q_next_eval = self.eval_net(nextstate_batch)
        #max_action_index = Q_next_eval.max(1)[1].view(-1, 1)
        Q_target = (reward_batch + (torch.logical_not(done_batch))*self.gamma*(Q_next.max(1)[0]).view(-1,1)).float()
        loss = self.loss_fun(Q_eval, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
        #     target_param.data.copy_(target_param.data * (1-self.updatestep) + param.data * self.updatestep)


        # if self.epsilon >= 0.1:
        #     self.epsilon = self.epsilon*0.999

    def save_parameters(self, save_path):
        torch.save(self.target_net.state_dict(), save_path)

    def load_parameters(self, load_path):
        tmp = torch.load(load_path, map_location=device)
        self.target_net.load_state_dict(tmp)
        self.eval_net.load_state_dict(tmp)





'''
下面是多版本结合的DQN,但是效果差，所以放弃使用
'''

'''
这里使用了Resnet18 dueling DQN
'''
class Resnet18(nn.Module):
    def __init__(self, inch, outdim):
        super(Resnet18,self).__init__()
        self.inchannel = inch
        self.outputdim = outdim


        self.resnet = resnet_dict["resnet18"](pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels=inch, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 256)
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),

        )
        self.outlayerA = nn.Linear(256, outdim)
        self.outlayerV = nn.Linear(256, 1)

    def forward(self, input):
        input = self.resnet(input)
        input = self.outputlayer(input)
        a = self.outlayerA(input)
        v = self.outlayerV(input)
        return v+(a-torch.mean(a))

'''
'VideoPinball-ramNoFrameskip-v4' 的单独网络，因为其输入为一维向量，不是图片
使用的dueling DQN
'''
class NNnet(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(NNnet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(inputdim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),

        )
        self.outlayerA = nn.Linear(256, outputdim)
        self.outlayerV = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer(x)
        a = self.outlayerA(x)
        v = self.outlayerV(x)
        return v+(a-torch.mean(a))

'''
这里是对priority buffer 的实现
'''
class SumTree(object):
    def __init__(self, buffer_size):
        self.data_pointer = 0
        self.size = buffer_size
        self.sum_tree = np.zeros(2*buffer_size-1)
        self.transitions = np.zeros(buffer_size, dtype=object)

    def add_transition(self, priority, transition):
        treedind = self.data_pointer+self.size-1
        self.transitions[self.data_pointer] = transition
        self.update(treedind, priority)
        self.data_pointer = (self.data_pointer+1) % self.size

    def update(self, treeind, priority):
        change = priority-self.sum_tree[treeind]
        self.sum_tree[treeind] = priority
        while treeind!=0:
            treeind = (treeind-1)//2
            self.sum_tree[treeind] += change

    def get_transition(self, priority):
        parent = 0
        while True:
            leftchild = 2*parent+1
            rightchild = leftchild+1
            if leftchild>=2*self.size-1:
                tree_ind = parent
                break
            else:
                if priority<=self.sum_tree[leftchild]:
                    parent = leftchild
                else:
                    priority = priority - self.sum_tree[leftchild]
                    parent = rightchild
        transition_ind = tree_ind - self.size + 1
        return tree_ind, self.sum_tree[tree_ind], self.transitions[transition_ind]

    def root(self):
        return self.sum_tree[0]

class PriorityBuffer(object):
    def __init__(self, buffer_size, batch_size):
        self.sumTree = SumTree(buffer_size)
        self.max_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = 0.01
        self.alpha = 0.3
        self.beta = 0.2
        self.beta_incrasing = 0.0002
        self.abs_err_upper = 1

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        priority = np.max(self.sumTree.sum_tree[-self.max_size:])
        if priority == 0:
            priority = self.abs_err_upper
        self.sumTree.add_transition(priority, transition)  # 添加到buffer中去
    def get_batches(self):
        simple_ind = np.empty((self.batch_size,), dtype=np.int32)
        sample_batch = np.zeros(self.batch_size, dtype=object)
        weight = np.empty((self.batch_size, 1))
        segment = self.sumTree.root()/self.batch_size
        self.beta = np.min([1., self.beta+self.beta_incrasing])
        min_p = np.min(self.sumTree.sum_tree[-self.max_size:])/self.sumTree.root()
        for i in range(self.batch_size):
            l, r = segment*i, segment*(i+1)
            pri = np.random.uniform(l,r)
            tree_ind, priority, transition = self.sumTree.get_transition(pri)
            probability = priority/self.sumTree.root()
            weight[i, 0] = np.power(probability/min_p, -self.beta)
            simple_ind[i] = tree_ind
            sample_batch[i] = transition

        return simple_ind, sample_batch, weight

    def batch_update(self, tree_indexs, abs_error):
        abs_error = abs_error+self.epsilon
        abs_error = np.minimum(abs_error, self.abs_err_upper)
        prioritys = np.power(abs_error,self.alpha)
        for tree, pr in zip(tree_indexs, prioritys):
            self.sumTree.update(tree, pr)

class myloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y,w):
        return torch.mean(w*torch.pow((x-y),2))

'''
结合了double DQN, dueling DQN 和 priority buffer 以及 resent18， 但是效果很差
'''
class ValueBasedModel_bad(object):

    def __init__(self, buffer_size, batch_size, action_dim, modelName, lr, updatestep, inchannel=1, inputdim=128):
        self.batch_size = batch_size
        self.replayBuffer = PriorityBuffer(buffer_size, batch_size)
        self.epsilon_start = 1.0
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        self.epsi_decay = 1000000
        self.gamma = 0.99
        self.action_dim = action_dim
        self.inchannel = inchannel
        self.name = modelName
        self.updatestep = updatestep
        self.inputdim = inputdim

        self.eval_net, self.target_net = self.build_net()
        self.eval_net.to(device)
        self.target_net.to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        self.loss_fun = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr,eps = 1.5e-4)

    def build_net(self):
        if self.name == 'VideoPinball-ramNoFrameskip-v4':
            net1 = NNnet(self.inputdim, self.action_dim)
            net2 = NNnet(self.inputdim, self.action_dim)
            return net1, net2

        else:
            net1 = Resnet18(self.inchannel, self.action_dim)
            net2 = Resnet18(self.inchannel, self.action_dim)
            return net1, net2


    def get_action(self, obs, mode=True):
        '''
        :param obs: 已经处理好的形状
        :param mode: 是否探索
        :return: action
        '''
        q_val = self.target_net(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
        if mode:
            self.epsilon -= (self.epsilon_start-self.epsilon_end)/self.epsi_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)
            if random.random() < self.epsilon:
                return np.random.randint(self.action_dim)
            else:
                return np.argmax(q_val)
        else:
            if random.random() < 0.05:
                return np.random.randint(self.action_dim)
            else:
                return np.argmax(q_val)
    def train(self):

        sample_ind, sample ,weight= self.replayBuffer.get_batches()

        state_batch = torch.FloatTensor(np.array([_[0] for _ in sample])).to(device)
        action_batch = torch.LongTensor(np.array([_[1] for _ in sample])).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor(np.array([_[2] for _ in sample])).view(-1, 1).to(device)
        nextstate_batch = torch.FloatTensor(np.array([_[3] for _ in sample])).to(device)
        done_batch = torch.FloatTensor(np.array([_[4] for _ in sample])).view(-1, 1).to(device)
        weight = torch.tensor(weight).view(self.batch_size, 1).view(-1, 1).to(device)
        Q_eval = self.eval_net(state_batch).gather(1, action_batch)
        Q_next = self.target_net(nextstate_batch).detach()
        Q_next_eval = self.eval_net(nextstate_batch)
        max_action_index = Q_next_eval.max(1)[1].view(-1, 1)
        Q_target = (reward_batch + (torch.logical_not(done_batch))*self.gamma*Q_next.gather(1, max_action_index).view(-1,1)).float()
        Qval = Q_eval.cpu().detach().numpy()
        Qtar = Q_target.cpu().numpy()
        abs_errs = np.abs(Qval, Qtar)
        self.replayBuffer.batch_update(sample_ind, abs_errs)
        loss = self.loss_fun(Q_eval, Q_target, weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_parameters(self, save_path):
        torch.save(self.target_net.state_dict(), save_path)

    def load_parameters(self, load_path):
        tmp = torch.load(load_path, map_location=device)
        self.target_net.load_state_dict(tmp)
        self.eval_net.load_state_dict(tmp)




