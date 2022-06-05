import gym
import numpy as np
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
import time


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)

# 建立一个网络
class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64, output_dim),


        )

    def forward(self, input):
        output = self.layer1(input)
        return output


class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
        )
        self.layer_a = nn.Sequential(
            nn.Linear(64,output_dim)
        )
        self.layer_v = nn.Sequential(
            nn.Linear(64,1)
        )

    def forward(self, input):
        output1 = self.layer1(input)
        output_a = self.layer_a(output1)
        output_v = self.layer_v(output1)
        output = output_v + (output_a - torch.mean(output_a))
        return output

# 建立DQN网络

class DQN(object):
    def __init__(self, input_dim,  output_dim, NetworkType):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.NetworkType = NetworkType  # 是否使用double DQN
        # replay buffer 设置
        self.BufferSize = 10000
        self.replayBuffer = np.zeros(self.BufferSize, dtype=object)
        self.currentpointer = 0  # 记录当前可以存放的位置
        self.batchSize = 1024
        # gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 创建两个网络
        self.evalNet = None
        self.targetNet = None
        self.buildNetwork()

        self.targetNet.load_state_dict(self.evalNet.state_dict())
        self.loss_fun = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.evalNet.parameters(), lr=0.0001)

        self.updateinterval = 20  # 每训练30次更新一次目标网络
        self.epsilon = 1  # 探索的概率
        self.gamma = 0.99
        self.learntimes = 0





    def buildNetwork(self):

        #DQN or double DQN
        if self.NetworkType == 0 or self.NetworkType == 1:
            self.evalNet = Network(self.input_dim, self.output_dim).to(self.device)
            self.targetNet = Network(self.input_dim, self.output_dim).to(self.device)

        # Dueling DQN or Dueling double DQN
        elif self.NetworkType == 2 or self.NetworkType == 3:
            self.evalNet = DuelingNet(self.input_dim, self.output_dim).to(self.device)
            self.targetNet = DuelingNet(self.input_dim, self.output_dim).to(self.device)






    # 根据状态获取下一个动作, 带探索的 或者 贪心的
    def get_action(self, obs, mode):

        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        q_val = self.targetNet(obs).detach().cpu().numpy()
        if mode:
            if random.random() < self.epsilon:
                return np.random.randint(self.output_dim)
            else:
                return np.argmax(q_val)
        else:
            return np.argmax(q_val)

     # 将数据存存入经验池 tans = (s, a, r, s1, d)
    def push_transition(self, trans):
        self.currentpointer = self.currentpointer % self.BufferSize
        self.replayBuffer[self.currentpointer] = trans
        self.currentpointer += 1

    # 从经验池中取出一个batch
    def get_traindata(self):
        Batch = np.random.choice(self.replayBuffer, self.batchSize, replace=False)
        obsBatch = []
        actionBatch = []
        rewardBatch = []
        nextObsBatch = []
        doneBatch = []

        for i in range(self.batchSize):
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

        # 获取一批数据
        obsBatch, actionBatch, rewardBatch, nextObsBatch, doneBatch = self.get_traindata()
        obsBatch = torch.from_numpy(obsBatch).float().view(-1, 2).to(self.device)
        actionBatch = torch.LongTensor(actionBatch).view(-1, 1).to(self.device)
        rewardBatch = torch.from_numpy(rewardBatch).view(-1, 1).to(self.device)
        nextObsBatch = torch.from_numpy(nextObsBatch).float().view(-1, 2).to(self.device)
        doneBatch = torch.from_numpy(doneBatch).view(-1, 1).to(self.device)

        q_eval = self.evalNet(obsBatch).gather(1, actionBatch)
        q_next = self.targetNet(nextObsBatch).detach()

        # double DQN or dueling double DQN
        if self.NetworkType == 1 or self.NetworkType == 3:
            q_next_eval = self.evalNet(nextObsBatch).detach()
            max_action_indx = q_next_eval.max(1)[1].view(-1,1)
            q_target = (rewardBatch + (torch.logical_not(doneBatch))*self.gamma*q_next.gather(1, max_action_indx).view(-1, 1)).float()

        # Dueling DQN or DQN
        else:
            q_target = (rewardBatch + (torch.logical_not(doneBatch))*self.gamma*(q_next.max(1)[0]).view(-1, 1)).float()

        loss = self.loss_fun(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





        if self.learntimes % self.updateinterval == 0:
            #self.targetNet.load_state_dict(self.evalNet.state_dict())
            for target_param, param in zip(self.targetNet.parameters(), self.evalNet.parameters()):
                target_param.data.copy_(target_param.data * 0.98 + param.data * 0.02)



        self.learntimes += 1
        self.epsilon = self.epsilon*0.99  # 下降




def testNetwork(parameterpath, type):
    environment = gym.make('MountainCar-v0')

    testModel = DQN(input_dim=2, output_dim=3, NetworkType=type)
    testModel.targetNet.load_state_dict(torch.load(parameterpath))

    rewardList = []
    for i in range(100):
        obs = environment.reset()
        totalReward = 0
        while True:
            #environment.render()
            action = testModel.get_action(obs, False)

            nextobs, reward, done, info = environment.step(action)
            totalReward += reward
            obs = nextobs
            if done:
                rewardList.append(totalReward)

                break
    environment.close()

    return rewardList





def drawfig(reward, name):


    plt.plot(list(range(len(reward))), reward)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(name)
    plt.savefig('imgs/'+name+'.png')
    plt.close()






def training(type):

    environment = gym.make('MountainCar-v0')
    maxEpisodes = 600
    DQNModel = DQN(input_dim=2, output_dim=3, NetworkType=type)
    step = 0
    rewardlist = []
    maxReward = -300
    for episode in range(maxEpisodes):
        observation = environment.reset()
        totalreward = 0
        while True:
            step += 1
            action = DQNModel.get_action(observation, True)
            nextobservation, reward, done, info = environment.step(action)
            totalreward += reward
            reward = abs(nextobservation[0]+0.5)

            transtion = (observation, action, reward, nextobservation, done)
            DQNModel.push_transition(transtion)

            observation = nextobservation

            if step > DQNModel.BufferSize+1:  # 存满后开始训练
                DQNModel.train()

            if done:
                rewardlist.append(totalreward)
                print('type:', type ,'episode:', episode, 'totalreward', totalreward)
                if (totalreward > maxReward and totalreward > -125) or totalreward > -90:
                    currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    savepath = 'parameters/' + str(type) + '_'+currentTime + "_"+ str(episode) + str(totalreward) + '.pkl'
                    torch.save(DQNModel.targetNet.state_dict(), savepath)
                    if totalreward>maxReward:
                        maxReward = totalreward
                break

    environment.close()
    return rewardlist


def main():
    set_seed(2)
    DQN_Reward = training(0)
    Double_DQN_Reward = training(1)
    Dueling_DQN_Reward = training(2)
    Dueling_Double_DQN_Reward = training(3)


    drawfig(DQN_Reward, 'DQN_Reward')
    drawfig(Double_DQN_Reward,  'Double_DQN_Reward')
    drawfig(Dueling_DQN_Reward,  'Dueling_DQN_Reward')
    drawfig(Dueling_Double_DQN_Reward,  'Dueling_Double_DQN_Reward')

def testmain():
    set_seed(2)
    DQNparameter = 'parameters/DQN.pkl'
    Double_DQNparameter = 'parameters/DoubleDQN.pkl'
    Dueling_DQNparameter = 'parameters/DuelingDQN.pkl'
    Double_Dueling_DQNparameter = 'parameters/Double_Dueling_DQN.pkl'

    DQN_reward = testNetwork(DQNparameter,0)
    Double_DQN_reward = testNetwork(Double_DQNparameter,1)
    Dueling_DQN_reward = testNetwork(Dueling_DQNparameter,2)
    Double_Dueling_DQN_reward = testNetwork(Double_Dueling_DQNparameter, 3)

    lenth = len(DQN_reward)

    plt.figure(figsize=(20, 10), dpi=90)
    plt.plot(list(range(lenth)), DQN_reward, label='DQN')
    plt.plot(list(range(lenth)), Double_DQN_reward, label='Double DQN')
    plt.plot(list(range(lenth)), Dueling_DQN_reward, label='Dueling DQN')
    plt.plot(list(range(lenth)), Double_Dueling_DQN_reward, label='Dueling&Double DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(loc = 1)
    plt.savefig('imgs/testResult.png')
    plt.close()

    print("DQN average: ", sum(DQN_reward)/lenth)
    print("Double DQN average: ", sum(Double_DQN_reward) / lenth)
    print("Dueling DQN average: ", sum(Dueling_DQN_reward) / lenth)
    print("Dueling&Double DQN average: ", sum(Double_Dueling_DQN_reward) / lenth)




if __name__ == '__main__':

    # main() is to train the network
    # testmain() is to test trained network

    # main()
    testmain()
