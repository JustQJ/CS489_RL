import time

import gym
import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

# share the
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# A3C network
class A3CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(A3CNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.actorNet = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),

        )


        self.actor_mu = nn.Sequential(
            nn.Linear(64, a_dim),
            nn.Tanh()
        )
        self.actor_sigma = nn.Sequential(
            nn.Linear(64, a_dim),
            nn.Softplus()
        )


        self.criticNet = nn.RNN(
            input_size=3,
            hidden_size=128,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(128, 1)



    def forward(self, input, h_state):

        a = self.actorNet(input)
        a_mu = 2 * self.actor_mu(a)
        a_sigma = self.actor_sigma(a) + 0.001

        # reshape the input
        input1 = input[None,:,:] #将这个reshape成 1*200*3
        out, h_state = self.criticNet(input1, h_state)
        v = []
        for i in range(out.size(1)):
            v.append(self.out(out[:, i, :]))




        return a_mu, a_sigma, torch.stack(v,dim=1), h_state

    def choose_action(self, state, test = False):
        h_state = None
        action_mu, action_sigma, v, h_state = self.forward(torch.FloatTensor(np.array(state)).view(-1, self.s_dim),h_state)
        h_state = h_state.data

        # 从正态分布中抽取动作
        if test:
            return action_mu.data.numpy()
        action = torch.normal(action_mu.view(1, ).data, action_sigma.view(1, ).data).numpy()
        return action

    def loss_func(self, s, a, R_t):
        h_state = None
        self.train()
        action_mu, action_sigma, values, h_state = self.forward(s,h_state)
        h_state = h_state.data
        values.view(-1,1)

        advantages = R_t - values

        # critic网络的loss

        critic_loss = advantages.pow(2)

        # actor 网络的loss
        distributions = Normal(action_mu, action_sigma)  # 计算每个动作的概率分布
        log_prob = distributions.log_prob(a)  # 计算每个动作的log值
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(distributions.scale)

        actor_loss = -(log_prob * (advantages.detach()) + 0.005 * entropy)  # 获得loss函数
        return (critic_loss + actor_loss).mean()

# worker
class worker(mp.Process):
    def __init__(self,  name,s_dim,a_dim, gamma ,share_optimizer, globalNet, globalmaxEpisodes, shareCurrentEpisode, shareQueue):
        super(worker, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.name = name
        self.globalNet = globalNet
        self.share_optimizer = share_optimizer
        self.localNet = A3CNet(self.s_dim,self.a_dim)

        self.globalmaxEpisodes = globalmaxEpisodes
        self.shareCurrentEpisode = shareCurrentEpisode
        self.shareQueue = shareQueue
        self.gamma = gamma

        self.env = gym.make('Pendulum-v1')


    def training(self, episode_s, episode_a, episode_r, final_s, done):
        # 计算R
        h_state = None
        R_t = []
        s_val = 0
        if not done:
            action_mu, action_sigma, v, h_state =self.localNet(torch.from_numpy(np.array(final_s)).view(-1, 3), h_state)
            h_state = h_state.data

            s_val = v.data.numpy()[0][0][0]



        # 逆序遍历计算每个state对于的值
        for r in episode_r[::-1]:
            s_val = r + self.gamma * s_val
            R_t.append(s_val)
        R_t.reverse()

        # 求loss
        loss = self.localNet.loss_func(torch.FloatTensor(np.array(episode_s)).view(-1, self.s_dim),
                                     torch.FloatTensor(np.array(episode_a)).view(-1, self.a_dim),
                                     torch.FloatTensor(np.array(R_t)).view(-1, 1))
        # self.actorOptimizer.zero_grad()
        # 更新全局参数


        self.share_optimizer.zero_grad()
        loss.backward()
        for thispara1, globpara1 in zip(self.localNet.parameters(), self.globalNet.parameters()):
            globpara1._grad = thispara1.grad
        self.share_optimizer.step()
        self.localNet.load_state_dict(self.globalNet.state_dict())



    def run(self):

        myepisode = 0
        while self.shareCurrentEpisode.value < self.globalmaxEpisodes:

            states = []
            actions = []
            rewards = []
            state = self.env.reset()

            step = 0
            totalreward = 0
            while True:
                # self.env.render()
                step += 1
                action = self.localNet.choose_action(state)
                next_state, reward, done, _ = self.env.step(action.clip(-2, 2))
                states.append(state)
                actions.append(action)
                rewards.append((reward+8.0)/8.0)
                totalreward += reward
                if done or step >= 200:
                    self.training(states, actions, rewards, next_state, done)
                    break

                #elif step%10 == 0:
                    #self.training(states,actions,rewards,next_state,done)
                state = next_state

            print(self.name, " ", myepisode, " ", self.shareCurrentEpisode.value, " ", step, " ", totalreward)

            # 通过调整这个条件，可以保存训练参数，这里不保存
            if (totalreward > 0):
                currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                savepath1 = 'a3cparamaters/' + self.name + '_' + currentTime + "_" + str(self.shareCurrentEpisode.value) + "_" + str(
                    totalreward) + '.pkl'
                torch.save(self.localNet.state_dict(), savepath1)


            myepisode += 1
            with self.shareCurrentEpisode.get_lock():
                self.shareCurrentEpisode.value += 1
            self.shareQueue.put(totalreward)
        self.shareQueue.put(None)




def main():
    env1 = gym.make('Pendulum-v1')
    StateDim = env1.observation_space.shape[0]
    ActionDim = env1.action_space.shape[0]
    globalNet = A3CNet(StateDim,ActionDim)
    globalNet.share_memory()
    share_optimizer = SharedAdam(globalNet.parameters(),lr=0.0001)
    MaxEpisodes = 7000
    shareCurrentEpisode = mp.Value('i', 0)
    shareQueue = mp.Queue()
    gamma = 0.9

    worker1 = worker("worker1",StateDim,ActionDim,gamma,share_optimizer,globalNet,MaxEpisodes,shareCurrentEpisode,shareQueue)
    worker2 = worker("worker2",StateDim,ActionDim,gamma,share_optimizer,globalNet,MaxEpisodes,shareCurrentEpisode,shareQueue)
    worker3 = worker("worker3",StateDim,ActionDim,gamma,share_optimizer,globalNet,MaxEpisodes,shareCurrentEpisode,shareQueue)
    worker4 = worker("worker4",StateDim,ActionDim,gamma,share_optimizer,globalNet,MaxEpisodes,shareCurrentEpisode,shareQueue)

    worker4.start()
    worker3.start()
    worker2.start()
    worker1.start()

    rewardlist = []
    while True:
        r = shareQueue.get()
        if r is not None:
            rewardlist.append(r)
        else:
            break
    worker4.join()
    worker3.join()
    worker2.join()
    worker1.join()

    re = 0
    rewardlist1 = []
    for item in rewardlist:
        if re == 0:
            re = item
        else:
            re = re*0.95+item*0.05
        rewardlist1.append(re)

    drawfig(rewardlist1)


def drawfig(rewardlist):
    plt.plot(list(range(len(rewardlist))), rewardlist)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('A3C')
    plt.savefig('imgs/A3C.png')



def test():
    env = gym.make('Pendulum-v1')
    StateDim = env.observation_space.shape[0]
    ActionDim = env.action_space.shape[0]
    maxEpisodes = 100
    maxstep = 200
    A3C = A3CNet(StateDim, ActionDim)
    A3C.load_state_dict(torch.load('worker4_2022_05_03_09_20_44_5673_-0.632508706081215.pkl'))
    totalreward = 0
    for i in range(maxEpisodes):
        s = env.reset()
        reward = 0
        step = 0
        while True:
            #env.render()
            step += 1
            a = A3C.choose_action(s, True)
            s_, r, d, _ = env.step(a)
            reward += r
            if step > maxstep:
                break

            s = s_
        totalreward += reward
    print("the average reward of 100 episodes: ", totalreward / maxEpisodes)


if __name__ == '__main__':


    # main is to train the model
    #main()

    # test is to test the model
    test()