'''
this file is to test the trained model
'''
import os.path
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from valueBased import ValueBasedModel
from policyBased import PolicyBasedModel
from train import obsPreprocessing


def drawtestresult(rewards, gameName):

    midreward = sum(rewards)/len(rewards)
    midrewards = [midreward for _ in range(len(rewards))]
    x = list(range(len(rewards)))
    plt.plot(x, rewards, c='black')
    plt.plot(x, midrewards, c='red', lw=2, linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(gameName)
    plt.show()
    plt.close()

def Continuous(env, model, maxEpisode, gameName):

    rewards = []
    for episode in range(maxEpisode):
        s = env.reset()
        totalreward = 0
        while True:
            #env.render()
            a = model.get_action(s, False)
            s1, r, d, info = env.step(a)
            totalreward += r
            if d:
                rewards.append(totalreward)
                print("game: ", gameName, " episode: ", episode+1, " reward: ", totalreward)
                break

            s = s1
    print("---------------------------------------------------------------------------------------------------")
    print(gameName, " avgreward: ", sum(rewards) / len(rewards))
    print("---------------------------------------------------------------------------------------------------")
    return rewards


def Discrete(envs, model, evalNumber, gameName):

    rewards = []

    if gameName == 'VideoPinball-ramNoFrameskip-v4':
        for episode in range(evalNumber):
            s = envs.reset()
            thisreward = 0
            while True:

                a = model.get_action(s, False)
                s1, r, d, info = envs.step(a)
                thisreward += r
                if d:
                    rewards.append(thisreward)
                    print("game: ", gameName, " episode: ", episode+1, " reward: ", thisreward)
                    break
                s = s1
    else:
        for episode in range(evalNumber):
            s = envs.reset()
            state_buffer = []
            s = obsPreprocessing(s)
            for j in range(4):
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
                    print("game: ", gameName, " episode: ", episode+1, " reward: ", thisreward)
                    break
                s = s1
    print("---------------------------------------------------------------------------------------------------")
    print(gameName, " avgreward: ", sum(rewards) / len(rewards))
    print("---------------------------------------------------------------------------------------------------")

    return rewards


def main(gameindex):

    Atarienvs = ['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4',
                 'BoxingNoFrameskip-v4']
    MuJuCoenvs = ['Hopper-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Ant-v2']
    antpath = [os.path.join('parameters','policyBased','Ant-v2',
                            '2022_05_26_23_26_53_2556_3236.896766060843_actor_.pkl'),
               os.path.join('parameters','policyBased','Ant-v2',
                            '2022_05_26_23_26_53_2556_3236.896766060843_critic1_.pkl'),
               os.path.join('parameters', 'policyBased', 'Ant-v2',
                            '2022_05_26_23_26_53_2556_3236.896766060843_critic2_.pkl')
               ]
    humanoidpath = [ os.path.join('parameters', 'policyBased', 'Humanoid-v2',
                                  '2022_05_20_00_42_16_221_316.4604061654624_actor_.pkl'),
                     os.path.join('parameters', 'policyBased', 'Humanoid-v2',
                                  '2022_05_20_00_42_16_221_316.4604061654624_critic1_.pkl'),
                     os.path.join('parameters', 'policyBased', 'Humanoid-v2',
                                  '2022_05_20_00_42_16_221_316.4604061654624_critic2_.pkl')
                    ]
    halfcheetahpath = [
                       os.path.join('parameters', 'policyBased', 'HalfCheetah-v2',
                                    '2022_05_20_13_00_32_886_5526.7048092840905_actor_.pkl'),
                       os.path.join('parameters', 'policyBased', 'HalfCheetah-v2',
                                    '2022_05_20_13_00_32_886_5526.7048092840905_critic1_.pkl'),
                       os.path.join('parameters', 'policyBased', 'HalfCheetah-v2',
                                    '2022_05_20_13_00_32_886_5526.7048092840905_critic2_.pkl')
                       ]
    hopperpath = [os.path.join('parameters', 'policyBased', 'Hopper-v2',
                               '2022_05_27_02_37_41_2970_3414.2628856193555_actor_.pkl'),
                  os.path.join('parameters', 'policyBased', 'Hopper-v2',
                               '2022_05_27_02_37_41_2970_3414.2628856193555_critic1_.pkl'),
                  os.path.join('parameters', 'policyBased', 'Hopper-v2',
                               '2022_05_27_02_37_41_2970_3414.2628856193555_critic2_.pkl')
                  ]

    pongpath = os.path.join('parameters', 'valueBased', 'PongNoFrameskip-v4','2022_06_04_08_30_44_13150000_9.4.pkl')
    breakoutpath = os.path.join('parameters', 'valueBased','BreakoutNoFrameskip-v4', '2022_06_03_01_45_06_12650000_49.6.pkl')
    pinballpath = os.path.join('parameters', 'valueBased','VideoPinball-ramNoFrameskip-v4', '2022_06_01_14_52_24_500000_26972.0.pkl')
    boxingpath = ''

    parameterspaths = [pinballpath, breakoutpath, pongpath, boxingpath, hopperpath, halfcheetahpath, humanoidpath,
                       antpath]

    gamenames = ['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4',
                 'BoxingNoFrameskip-v4', 'Hopper-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Ant-v2']

    parameters = parameterspaths[gameindex]
    gameName = gamenames[gameindex]



    if gameName in MuJuCoenvs:
        actor_mid_dim = [256, 256, 256, 256]
        critic_mid_dim = [256, 256, 256, 512]
        env = gym.make(gameName)
        action_dim = env.action_space.shape[0]
        actiom_max = env.action_space.high
        state_dim = env.observation_space.shape[0]
        maxeposide = 50
        model = PolicyBasedModel(state_dim=state_dim, action_dim=action_dim, max_action=actiom_max[0], buffersize=5,
                                  batchsize=1, critic_lr=0.01, actor_lr=0.001, updateinterval=2,
                                  actor_mid_dim=actor_mid_dim[gameindex-4], critic_mid_dim=critic_mid_dim[gameindex-4])
        model.load_parameters(parameters)
        rewards = Continuous(env, model, maxeposide, gameName)
        env.close()
        drawtestresult(rewards, gameName)



    elif gameName in Atarienvs:
        env = gym.make(gameName)
        actionDim = env.action_space.n
        maxepisode = 50
        model = ValueBasedModel(buffer_size=5, batch_size=1, action_dim=actionDim,modelName=gameName,
                                lr=0.1, updatestep=100, inchannel=4)
        model.load_parameters(parameters)
        rewards = Discrete(env, model, maxepisode, gameName)
        env.close()
        drawtestresult(rewards, gameName)



if __name__ == "__main__":
    args = sys.argv
    envs = ['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4',
            'Hopper-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Ant-v2']
    if len(args) != 3 or args[1] != '--env_name' or args[2] not in envs:
        print("input error!")
    elif args[2] == 'BoxingNoFrameskip-v4':
        print("Don't train a model for it!")
    else:
        gameindex = envs.index(args[2])
        main(gameindex)






