
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import time


#Sarsa Alogrithm
#一个三维数据Q记录所有的q值
#定义discount factor 和 最大迭代次数 epsilon-greedy的参数 alpha的比重

class Sarsa(object):
    def __init__(self, discount, maxIteration, epsilon, alpha) -> None:
        self.discount = discount
        self.maxIteration = maxIteration
        self.epsilon = epsilon
        self.alpha = alpha

        self.Q_Value = np.random.random((4,12,4)) #初始化
        self.Q_Value[3][11][0] = self.Q_Value[3][11][1]=self.Q_Value[3][11][2]=self.Q_Value[3][11][3]=0
        self.terminal = [3,11]

        self.actions = [[-1,0],[1,0],[0,-1],[0,1]]
        self.cliffstate = [[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10]]

    def findOptimalPath(self):

        iteration = 0
        cost = []
        while iteration < self.maxIteration:
            self.alpha = 0.99*self.alpha
            state = [3,0]
            nextstate = [0,0]
            
            tempcost = 0
            action = self.getaction(state)
            while state !=self.terminal:
                nextstate[0] = max(0,min(3,state[0]+self.actions[action][0]))
                nextstate[1] = max(0,min(11,state[1]+self.actions[action][1]))
                
                reward = -1
                if nextstate in self.cliffstate: #判断是否是特殊状态
                    reward = -100
                    nextstate[0] = 3
                    nextstate[1] = 0

                tempcost += reward
                nextaction = self.getaction(nextstate)

                self.Q_Value[state[0]][state[1]][action] = (1-self.alpha)*self.Q_Value[state[0]][state[1]][action] + self.alpha*(reward+self.discount*self.Q_Value[nextstate[0]][nextstate[1]][nextaction])
                
                state[0] = nextstate[0]
                state[1] = nextstate[1]
                action = nextaction
            iteration += 1
            cost.append(tempcost)

        state = [3,0]
        actionlist = []
        realaction = ['up','down','left','right']
        while state!=self.terminal:
            action = np.argmax(self.Q_Value[state[0]][state[1]])
            actionlist.append(realaction[action])
            state[0] = max(0,min(3,state[0]+self.actions[action][0]))
            state[1] = max(0,min(11,state[1]+self.actions[action][1]))

            if state in self.cliffstate:
                state[0] = 3
                state[1] = 0
        
        return actionlist, cost

    #带探索的贪心策略
    def getaction(self,state):
        greedyAction = np.argmax(self.Q_Value[state[0]][state[1]]) 
        p = np.random.random()
        if p < 1-self.epsilon:
            return greedyAction
        else:
            return np.random.randint(4)



#Qlearning

class Q_learing(object):
    def __init__(self, discount, maxIteration, epsilon, alpha) -> None:
        self.discount = discount
        self.maxIteration = maxIteration
        self.epsilon = epsilon
        self.alpha = alpha

        self.Q_Value = np.random.random((4,12,4)) #初始化
        self.Q_Value[3][11][0] = self.Q_Value[3][11][1]=self.Q_Value[3][11][2]=self.Q_Value[3][11][3]=0
        self.terminal = [3,11]

        self.actions = [[-1,0],[1,0],[0,-1],[0,1]]
        self.cliffstate = [[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10]]

    def findOptimalPath(self):

        iteration = 0
        cost = [] #记录每一个episode的值
        while iteration < self.maxIteration:
            self.alpha = 0.99*self.alpha
            state = [3,0]
            nextstate = [0,0]
            tempcost = 0
            while state !=self.terminal:
                action = self.getaction(state)
                nextstate[0] = max(0,min(3,state[0]+self.actions[action][0]))
                nextstate[1] = max(0,min(11,state[1]+self.actions[action][1]))
                
                reward = -1
                if nextstate in self.cliffstate: #判断是否是特殊状态
                    reward = -100
                    nextstate[0] = 3
                    nextstate[1] = 0

                tempcost += reward 
                self.Q_Value[state[0]][state[1]][action] = (1-self.alpha)*self.Q_Value[state[0]][state[1]][action] + self.alpha*(reward+self.discount*np.max(self.Q_Value[nextstate[0]][nextstate[1]]))
                
                state[0] = nextstate[0]
                state[1] = nextstate[1]
               
            iteration += 1
            cost.append(tempcost)

        state = [3,0]
        realaction = ['up','down','left','right']
        actionlist = []
        while state!=self.terminal:
            action = np.argmax(self.Q_Value[state[0]][state[1]])
            actionlist.append(realaction[action])
            state[0] = max(0,min(3,state[0]+self.actions[action][0]))
            state[1] = max(0,min(11,state[1]+self.actions[action][1]))

            if state in self.cliffstate:
                state[0] = 3
                state[1] = 0
        
        return actionlist, cost

    def getaction(self,state):
        greedyAction = np.argmax(self.Q_Value[state[0]][state[1]]) 
        p = np.random.random()
        if p < 1-self.epsilon:
            return greedyAction
        else:
            return np.random.randint(4)





def main():

    np.random.seed(0)
    discount = 0.9
    maxIteration = 700
    epsilon = [0,0.05,0.1,0.15,0.2,0.25]
    alpha = 0.5
    SarsaRewardList = []
    Q_learningRewardList = []
    CostTime = []
    Sarsa_ActionList = []
    Q_learingActionList = []


    for i in range(len(epsilon)):
        t1 = time.time()
        SarsaModel = Sarsa(discount=discount, maxIteration=maxIteration,epsilon=epsilon[i],alpha=alpha)
        Sarsa_Action, Sarsa_Cost = SarsaModel.findOptimalPath()
        Q_learingModel = Q_learing(discount=discount, maxIteration=maxIteration,epsilon=epsilon[i],alpha=alpha)
        Q_learing_Action, Q_learing_Cost = Q_learingModel.findOptimalPath()
        t2 = time.time()

        CostTime.append(t2-t1)
        Sarsa_ActionList.append(Sarsa_Action)
        Q_learingActionList.append(Q_learing_Action)
        SarsaRewardList.append(Sarsa_Cost)
        Q_learningRewardList.append(Q_learing_Cost)


    #画图

    subfig = [231,232,233,234,235,236]
    plt.figure(figsize=(20,10),dpi=90)
    for i in range(len(epsilon)):
        plt.subplot(subfig[i])
        plt.plot(list(range(maxIteration)),SarsaRewardList[i],label="Sarsa")
        plt.plot(list(range(maxIteration)),Q_learningRewardList[i],label="Q_learning")
        plt.xlabel("Episodes")
        plt.ylabel("Path Cost")
        plt.legend()
        plt.title("epsilon="+str(epsilon[i]))
    plt.savefig('imgs/costChanges.png')
    plt.close()


    plt.plot(epsilon, CostTime)
    plt.xlabel("Different Epsilons")
    plt.ylabel("Cost Time(s)")
    plt.savefig('imgs/CostTime.png')
    plt.close()

    #每一个不同参数下的actions
    print("Sarsa actions:")
    for i in range(len(epsilon)):
        print("epsilon="+str(epsilon[i])+": ",Sarsa_ActionList[i])
    print()
    print("Q_learning actions:")
    for i in range(len(epsilon)):
        print("epsilon="+str(epsilon[i])+": ",Q_learingActionList[i])



if __name__ == "__main__":
    main()

