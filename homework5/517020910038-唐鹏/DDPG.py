import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import time



# actor网络
class ActorNet(nn.Module):
    def __init__(self,s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim)
        )
        self.actor_mu = nn.Tanh()


    def forward(self, input):
        a = self.actor(input)
        a_mu = 2 * self.actor_mu(a)  # 连续分布，计算动作的均值和方差, 动作（-2，2）
        return a_mu


# critic网络, 输入是action + state
class CriticNet(nn.Module):
    def __init__(self, s_dim):
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(s_dim+1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, input):
        return self.critic(input)

# DDPG 网络

class DDPGNet(object):
    def __init__(self, s_dim, a_dim):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.gamma = 0.9

        self.critic = CriticNet(s_dim)
        self.critic_target = CriticNet(s_dim)

        self.actor = ActorNet(s_dim,a_dim)
        self.actor_target = ActorNet(s_dim,a_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.action_var = 0.5

        self.bathsize = 256
        self.BufferSize = 5000
        self.currentpointer = 0
        self.replayBuffer = np.zeros(self.BufferSize,dtype=object)

        self.loss_fun = torch.nn.MSELoss()



    def get_action(self, state, noise = False): # 如果是true，就不使用噪声

        action_mu = self.actor(torch.FloatTensor(np.array(state)).view(-1,self.s_dim))
        action = torch.normal(action_mu.view(1,).data, torch.from_numpy(np.array([self.action_var]))).numpy()

        if noise:
            return action_mu.view(1,).data.numpy()
        else:
            self.action_var= self.action_var*0.99
            return action

    def push_transition(self,trans):
        self.currentpointer = self.currentpointer % self.BufferSize
        self.replayBuffer[self.currentpointer] = trans
        self.currentpointer +=1

    def get_traindata(self):
        Batch = np.random.choice(self.replayBuffer,self.bathsize,replace=False)
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


    def train(self):
        obsBatch, actionBatch, rewardBatch, nextObsBatch, doneBatch = self.get_traindata()
        obsBatch = torch.FloatTensor(obsBatch).view(-1,self.s_dim)
        actionBatch = torch.FloatTensor(actionBatch).view(-1,1)
        rewardBatch = torch.FloatTensor(rewardBatch).view(-1,1)
        nextObsBatch = torch.FloatTensor(nextObsBatch).view(-1,self.s_dim)
        doneBatch = torch.from_numpy(doneBatch).view(-1, 1)

        # 更新critic 网络
        q_eval = self.critic(torch.cat((obsBatch,actionBatch),1))
        nextActionBatch = self.actor_target(nextObsBatch)
        q_next = self.critic_target(torch.cat((nextObsBatch, nextActionBatch),1)).detach()
        q_target = rewardBatch+(torch.logical_not(doneBatch))*self.gamma*q_next
        loss = self.loss_fun(q_eval,q_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # 更新actor网络
        # 这里就是最大化critic的值
        # 用actor计算state下的action
        actionpre = self.actor(obsBatch)
        # 用critic计算Q值
        q_pre = self.critic(torch.cat((obsBatch, actionpre), 1))
        loss1 = -q_pre.mean()
        self.actor_optimizer.zero_grad()
        loss1.backward()
        self.actor_optimizer.step()

        # 更新目标网络

        for target_critic_param, critic_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_critic_param.data.copy_(target_critic_param.data*0.9 + critic_param.data*0.1)

        for target_actor_param, actor_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_actor_param.data.copy_(target_actor_param.data*0.9+actor_param.data*0.1)



def drawfig(rewardlist):
    plt.plot(list(range(len(rewardlist))), rewardlist)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('DDPG')
    plt.savefig('imgs/DDPG.png')



def running():
    env = gym.make('Pendulum-v1')
    StateDim = env.observation_space.shape[0]
    ActionDim = env.action_space.shape[0]
    maxEpisodes = 2000
    maxstep = 400
    DDPG = DDPGNet(StateDim, ActionDim)
    totalstep = 0
    rewardlist = []

    for episode in range(maxEpisodes):
        obs = env.reset()
        totalreward = 0
        step = 0
        while True:
            #env.render()
            totalstep +=1
            step += 1
            action = DDPG.get_action(obs)
            nextobs, reward, done, _ = env.step(action)
            totalreward += reward
            transtion = (obs,action,reward,nextobs,done)
            DDPG.push_transition(transtion)

            if totalstep>DDPG.BufferSize+1:
                DDPG.train()

            if done or step>maxstep:

                print(episode, " ", totalreward)

                # 存储参数，-125为界
                if(totalreward>0):
                    currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    savepath1 = 'ddpgparamaters/' + "actor" + '_' + currentTime + "_" + str(episode) +"_"+ str(
                        totalreward) + '.pkl'
                    savepath2 = 'ddpgparamaters/' + "critic" + '_' + currentTime + "_" + str(episode) + "_" + str(
                        totalreward) + '.pkl'
                    torch.save(DDPG.actor.state_dict(), savepath1)
                    torch.save(DDPG.critic.state_dict(),savepath2)


                break
            obs = nextobs
        rewardlist.append(totalreward)

    re = 0
    rewardlist1 = []
    for item in rewardlist:
        if re == 0:
            re = item
        else:
            re = re * 0.95 + item * 0.05
        rewardlist1.append(re)

    drawfig(rewardlist1)


# 用来测试保存的参数
def test():
    env = gym.make('Pendulum-v1')
    StateDim = env.observation_space.shape[0]
    ActionDim = env.action_space.shape[0]
    maxEpisodes = 100
    maxstep = 200
    DDPG = DDPGNet(StateDim, ActionDim)
    DDPG.actor.load_state_dict(torch.load('actor_2022_04_30_22_09_53_1263_-0.22505500796893066.pkl'))
    totalreward = 0
    for i in range(maxEpisodes):
        s = env.reset()
        reward = 0
        step = 0
        while True:
            env.render()
            step+=1
            a = DDPG.get_action(s, True)
            s_, r, d, _ = env.step(a)
            reward+=r
            if step>maxstep:
                break

            s = s_
        totalreward += reward
    print("the average reward of 100 episodes: ", totalreward/maxEpisodes)





if __name__ == "__main__":

    # this function is to train
    #running()

    # this function is to test
    test()







