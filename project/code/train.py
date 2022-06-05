'''
this file is used to train the model for games
'''
import os

import gym
import numpy as np
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
import time
import cv2
from valueBased import ValueBasedModel
from policyBased import PolicyBasedModel

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)

'''
对图像进行加工，主要是atari游戏，210*160*3 -> 84*84
'''
def obsPreprocessing(observation):
    '''
    将彩色图装换为灰度图，这样更加简单，容易处理
    :param observation: 210*160*3 的rgb图像
    :return: 84*84 的灰度图
    '''
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
    return observation

def drawfigs(rewards, gameName):
    re = 0
    for i in range(len(rewards)):
        if i == 0:
            re = rewards[i]
        else:
            re = re * 0.98 + rewards[i] * 0.02
        rewards[i] = re

    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(gameName)
    plt.savefig(os.path.join('imgs', gameName+'.png'))
    plt.close()

'''
atari 游戏的 评估当前的模型
'''
def evalDiscretePolicy(model, gameName, trainsteps, channels):
    '''

    :param model: 模型
    :param gameName: 游戏名字
    :param trainsteps: 已经训练的步数
    :param channels: 图像的输入通道数
    :return: 输出5次的平均得分
    '''
    envs = gym.make(gameName)
    rewards = []
    evalNumber = 5

    if gameName == 'VideoPinball-ramNoFrameskip-v4':
        for i in range(evalNumber):
            s = envs.reset()
            thisreward = 0
            while True:
                a = model.get_action(s, False)
                s1, r, d, info = envs.step(a)
                thisreward += r
                if d:
                    rewards.append(thisreward)
                    break
                s = s1
    else:
        for i in range(evalNumber):
            s = envs.reset()
            state_buffer = []
            s = obsPreprocessing(s)
            for j in range(channels):
                state_buffer.append(s)

            s = np.array(state_buffer)
            thisreward = 0
            while True:
                a = model.get_action(s.reshape(1, s.shape[0], s.shape[1], s.shape[2]), False)
                s1, r, d, info = envs.step(a)
                s1 = obsPreprocessing(s1)
                state_buffer.pop(0)
                state_buffer.append(s1)
                s1 = np.array(state_buffer)
                thisreward += r
                if d:
                    rewards.append(thisreward)
                    break
                s = s1
    envs.close()

    print("---------------------------------------------------------------------------------------------------")
    print(gameName, " evaluation results after training ", trainsteps, " step:")
    print(rewards, " avgreward: ", sum(rewards) / len(rewards))
    print("---------------------------------------------------------------------------------------------------")

    return sum(rewards) / len(rewards)

'''
对atari游戏的模型进行训练
'''
def trainDiscrete(seed ,gameName, batchsize, maxEpisode, buffersize, lr, updatestep):
    '''
    :param seed: 是随机数种子
    :param gameName: 游戏的名字，
    :param batchsize: 批训练大小
    :param maxEpisode: 训练回合
    :param buffersize: 记忆库大小
    :param lr: 学习率
    :return:
    '''
    set_seed(seed)
    env = gym.make(gameName)
    actionDim = env.action_space.n
    updateinterval = 4
    evaluatestep = 50000
    target_update = 10000
    channels = 4
    trainModel = ValueBasedModel(buffer_size=buffersize, batch_size=batchsize, action_dim=actionDim,modelName=gameName, lr=lr, updatestep=updatestep, inchannel=channels)
    step = 0
    rewards = []
    maxreward = 0
    if gameName == 'VideoPinball-ramNoFrameskip-v4':
        for episode in range(maxEpisode):
            s = env.reset()
            totalreward = 0
            while True:
                step += 1
                a = trainModel.get_action(s)
                s1, r, d, info = env.step(a)
                totalreward += r
                trainModel.replayBuffer.push(s, a, r, s1, d)
                if step > buffersize + 1 and step % updateinterval == 0:
                    trainModel.train()
                # if d or step % 1000 == 0:
                #     print("game: ", gameName, " episode: ", episode, " step: ", step, " reward: ", totalreward)
                if step % target_update == 0:
                    trainModel.target_net.load_state_dict(trainModel.eval_net.state_dict())
                if step > buffersize + 1 and step % evaluatestep == 0:
                    evalreward = evalDiscretePolicy(trainModel, gameName, step, channels)
                    if evalreward > maxreward:
                        maxreward = evalreward
                        currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                        savepath = os.path.join('parameters', 'valueBased', gameName,
                                                currentTime + '_' + str(step) + '_' + str(evalreward) + '.pkl')
                        trainModel.save_parameters(savepath)
                if d:
                    print("game: ", gameName, " episode: ", episode, " step: ", step, " reward: ", totalreward)
                    break
                s = s1
            rewards.append(totalreward)
    else:
        for episode in range(maxEpisode):
            state_buffer = []  # 用4个状态叠加作为一个状态
            s = env.reset()
            s = obsPreprocessing(s)
            for i in range(channels):
                state_buffer.append(s)
            s = np.array(state_buffer)
            totalreward = 0
            while True:
                step += 1

                a = trainModel.get_action(s.reshape(1,s.shape[0], s.shape[1], s.shape[2]))
                s1, r, d, info = env.step(a)
                s1 = obsPreprocessing(s1)
                state_buffer.pop(0)
                state_buffer.append(s1)
                s1 = np.array(state_buffer)
                totalreward += r
                trainModel.replayBuffer.push(s, a, r, s1, d)

                if step > buffersize+1 and step % updateinterval == 0:
                    trainModel.train()
                if d or step % 200 == 0:
                    print("game: ", gameName, " episode: ", episode, " step: ", step, " reward: ", totalreward)
                if step % target_update == 0:
                    trainModel.target_net.load_state_dict(trainModel.eval_net.state_dict())
                if step > buffersize+1 and step % evaluatestep == 0:
                    evalreward = evalDiscretePolicy(trainModel, gameName, step, channels)
                    if evalreward > maxreward:
                        maxreward = evalreward
                        currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                        savepath = os.path.join('parameters', 'valueBased', gameName ,  currentTime + '_' + str(step) + '_' + str(evalreward) + '.pkl')
                        trainModel.save_parameters(savepath)


                if d:
                    # if totalreward > maxreward:
                    #     currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    #     savepath = os.path.join('parameters', 'valueBased', gameName ,  currentTime + '_' + str(episode)+ '_'+ str(totalreward) + '.pkl')
                    #     trainModel.save_parameters(savepath)
                    #     maxreward = totalreward

                    break

                s = s1
            rewards.append(totalreward)

    env.close()
    drawfigs(rewards, gameName)


'''
对MuJuCo模型进行训练
'''
def trainContinuous(gameName, batchsize, maxEpisode, critic_rl, actor_rl, updateinterval, actor_mid_dim, critic_mid_dim):
    '''
    :param gameName: 游戏名
    :param batchsize: 批训练大小
    :param maxEpisode: 训练回合
    :param critic_rl: critic网络学习率
    :param actor_rl: actor网络学习率
    :param actor_mid_dim: actor网络中间层数量
    :param critic_mid_dim: critic网络中间层数量
    :return:
    '''
    env = gym.make(gameName)
    action_dim = env.action_space.shape[0]
    actiom_max = env.action_space.high
    state_dim = env.observation_space.shape[0]
    buffersize = 100000
    trainModel = PolicyBasedModel(state_dim=state_dim, action_dim=action_dim, max_action=actiom_max[0], buffersize=buffersize,
                                  batchsize=batchsize, critic_lr=critic_rl, actor_lr=actor_rl, updateinterval=updateinterval,
                                  actor_mid_dim=actor_mid_dim, critic_mid_dim=critic_mid_dim)
    step = 0
    rewards = []
    maxreward = 0
    for episode in range(maxEpisode):
        s = env.reset()
        totalreward = 0
        while True:
            step += 1
            a = trainModel.get_action(s)
            s1, r, d, info = env.step(a)
            totalreward += r
            trainModel.replayBuffer.push_transition((s, a, r, s1, d))

            if step > buffersize+1:
                trainModel.train()
            if d:
                if episode+1 > 200 and totalreward>2000 and totalreward>maxreward :
                    currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    savepath = os.path.join('parameters', 'policyBased',
                                            gameName, currentTime + '_' + str(episode) + '_' + str(
                                                totalreward))
                    maxreward = totalreward
                    trainModel.save_parameters(savepath)
                print("game: ", gameName, " episode: ", episode, " step: ", step, " reward: ", totalreward)
                break

            s = s1
        rewards.append(totalreward)


    env.close()

    drawfigs(rewards,gameName)


def main():


    Atarienvs = ['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4',
                 'BoxingNoFrameskip-v4']
    batchsize1 = [256, 32, 32, 16]
    maxepisode1 = [500, 10000, 1000, 100]
    buffersize = [200000, 200000, 200000, 5000]
    lr = [3e-4, 0.0000625, 0.0000625, 1e-3]
    updatestep = [0.01, 0.02, 0.02, 0.02, 0.02]
    seed = 1
    for i in range(1,len(Atarienvs)):
        trainDiscrete(seed, Atarienvs[i], batchsize1[i], maxepisode1[i], buffersize[i] , lr[i], updatestep[i])



    MuJuCoenvs = ['Hopper-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Ant-v2']
    batchsize2 = [256, 256, 256, 256]
    maxepisode2 = [3000, 1000, 20000, 3000]
    critic_rl = [3e-4, 3e-4, 8e-4, 0.5e-4]
    actor_rl = [3e-4, 3e-4, 1e-3, 2e-4]
    updateinterval = [2, 2, 3, 4]
    actor_mid_dim = [256, 256, 512, 256]
    critic_mid_dim = [256, 256, 1024, 512]
    for i in range(len(MuJuCoenvs)):
        trainContinuous(MuJuCoenvs[i], batchsize2[i], maxepisode2[i], critic_rl[i], actor_rl[i],updateinterval[i],actor_mid_dim[i], critic_mid_dim[i])



if __name__ == "__main__":
    '''
    main() 函数会对8个游戏进行训练
    '''
    main()

