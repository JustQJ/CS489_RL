'''
this file is define the model of policybased algorithm TD3
'''
import torch
from torch import nn
import numpy as np
import random
from torch.nn import functional as F
from torchvision import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, max_action, mid_dim):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim,256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh()
        )

    def forward(self, input):
        action = self.max_action*self.net(input)
        return action

class Critic(nn.Module):
    def __init__(self, in_dim, mid_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 256),
            nn.ReLU(),
            nn.Linear(256,1)

        )

    def forward(self, input):

        val = self.net(input)
        return val


class ReplayBuffer(object):
    def __init__(self, buffersize, bathsize):
        self.size = buffersize
        self.bathsize = bathsize
        self.currentpointer = 0
        self.buffer = np.zeros(self.size, dtype=object)

    def push_transition(self, trans):
        self.currentpointer = self.currentpointer % self.size
        self.buffer[self.currentpointer] = trans
        self.currentpointer += 1

    def get_batch(self):
        Batch = np.random.choice(self.buffer, self.bathsize, replace=False)
        obsBatch = []
        actionBatch = []
        rewardBatch = []
        nextObsBatch = []
        doneBatch = []

        for i in range(self.bathsize):
            obsBatch.append(Batch[i][0])
            actionBatch.append(Batch[i][1])
            rewardBatch.append(Batch[i][2])
            nextObsBatch.append(Batch[i][3])
            doneBatch.append(Batch[i][4])

        obsBatch = np.array(obsBatch)
        actionBatch = np.array(actionBatch)
        rewardBatch = np.array(rewardBatch)
        nextObsBatch = np.array(nextObsBatch)
        doneBatch = np.array(doneBatch)

        return obsBatch, actionBatch, rewardBatch, nextObsBatch, doneBatch

'''
TD3
'''
class PolicyBasedModel(object):
    def __init__(self, state_dim, action_dim, max_action, buffersize, batchsize, critic_lr, actor_lr, updateinterval, actor_mid_dim, critic_mid_dim):

        self.eval_actor = Actor(state_dim,action_dim,max_action,actor_mid_dim).to(device)
        self.eval_critic1 = Critic(state_dim+action_dim, critic_mid_dim).to(device)
        self.eval_critic2 = Critic(state_dim+action_dim, critic_mid_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.eval_actor.parameters(), lr=actor_lr)
        self.eval_critic1_optimizer = torch.optim.Adam(self.eval_critic1.parameters(), lr=critic_lr)
        self.eval_critic2_optimizer = torch.optim.Adam(self.eval_critic2.parameters(), lr=critic_lr)

        self.target_actor = Actor(state_dim,action_dim,max_action,actor_mid_dim).to(device)
        self.target_critic1 = Critic(state_dim+action_dim, critic_mid_dim).to(device)
        self.target_critic2 = Critic(state_dim+action_dim, critic_mid_dim).to(device)
        self.target_actor.load_state_dict(self.eval_actor.state_dict())
        self.target_critic1.load_state_dict(self.eval_critic1.state_dict())
        self.target_critic2.load_state_dict(self.eval_critic2.state_dict())

        self.replayBuffer = ReplayBuffer(buffersize, batchsize)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.buffersize = buffersize
        self.batchsize = batchsize
        self.gamma = 0.99
        self.beta = 0.98
        self.sigma = 0.05
        self.action_var = 0.1
        self.clip = 0.5

        self.step = 0
        self.updateinterval = updateinterval

    def get_action(self, state, noise=True):

        with torch.no_grad():
            action_mu = np.squeeze(self.eval_actor(torch.FloatTensor(state).view(-1,self.state_dim).to(device)).cpu().data.numpy())


        if noise:
            return (action_mu+np.random.normal(0, self.max_action*self.action_var, size=self.action_dim)).clip(-self.max_action, self.max_action)
        else:
            return action_mu
    def train(self):


        obsBatch, actionBatch, rewardBatch, nextObsBatch, doneBatch = self.replayBuffer.get_batch()
        obsBatch = torch.FloatTensor(obsBatch).view(-1, self.state_dim).to(device)
        actionBatch = torch.FloatTensor(actionBatch).view(-1, self.action_dim).to(device)
        rewardBatch = torch.FloatTensor(rewardBatch).view(-1, 1).to(device)
        nextObsBatch = torch.FloatTensor(nextObsBatch).view(-1, self.state_dim).to(device)
        doneBatch = torch.from_numpy(doneBatch).view(-1, 1).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(actionBatch)*0.2).clamp(-self.clip, self.clip).to(device)

            next_action = (self.target_actor(nextObsBatch) + noise).clamp(-self.max_action, self.max_action)

            target1 = self.target_critic1(torch.cat((nextObsBatch,next_action),1))
            target2 = self.target_critic2(torch.cat((nextObsBatch, next_action), 1))
            target = rewardBatch + (torch.logical_not(doneBatch))*self.gamma*torch.min(target1, target2)

        eval1 = self.eval_critic1(torch.cat((obsBatch, actionBatch),1))
        eval2 = self.eval_critic2(torch.cat((obsBatch, actionBatch),1))

        loss = F.mse_loss(eval1, target) + F.mse_loss(eval2, target)


        self.eval_critic1_optimizer.zero_grad()
        self.eval_critic2_optimizer.zero_grad()
        loss.backward()
        self.eval_critic1_optimizer.step()
        self.eval_critic2_optimizer.step()

        self.step += 1
        if self.step % self.updateinterval == 0:

            actor_loss = -self.eval_critic1(torch.cat((obsBatch, self.eval_actor(obsBatch)), 1)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.eval_actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.beta*target_param.data + (1-self.beta)*param)

            for param, target_param in zip(self.eval_critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.beta*target_param.data + (1-self.beta)*param)
            for param, target_param in zip(self.eval_critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.beta*target_param.data + (1-self.beta)*param)

    def save_parameters(self, save_path):
        torch.save(self.eval_actor.state_dict(), save_path+'_actor_.pkl')
        torch.save(self.eval_critic1.state_dict(), save_path + '_critic1_.pkl')
        torch.save(self.eval_critic2.state_dict(), save_path + '_critic2_.pkl')

    def load_parameters(self, load_path):
        self.eval_actor.load_state_dict(torch.load(load_path[0], map_location=device))
        self.eval_critic1.load_state_dict(torch.load(load_path[1], map_location=device))
        self.eval_critic2.load_state_dict(torch.load(load_path[2], map_location=device))
        self.target_actor.load_state_dict(torch.load(load_path[0], map_location=device))
        self.target_critic1.load_state_dict(torch.load(load_path[1], map_location=device))
        self.target_critic2.load_state_dict(torch.load(load_path[2], map_location=device))













