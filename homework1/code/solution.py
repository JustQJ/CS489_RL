

from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import time
#目标：找到最优的策略



#定义一个类

#成员变量：每个状态对应的值，每个状态对应的策略，值迭代的终点条件和discount factor
#成员函数：policy Evaluation, policy Improvement, 和找到一个状态的最优策略的函数
class PolicyIteration(object):
    def __init__(self,discount):
        #初始化
        self.BasicAction = [[0,1],[1,0],[0,-1],[-1,0]]  #四个基本的动作
        self.action_convert = ["right","down","left","up"]
        self.Value = np.zeros([6,6]) #所有状态的初始价值全部为0
        self.error = 0.01
        self.discntFactor = discount
        self.reward = -1
        self.finalState = [[0,1],[5,5]]

        #记录迭代的次数和error
  
        self.changeserror = []

        #所有状态的初始动作为四个方向的随机动作
        self.Actions = np.empty([6,6],dtype = object)
        for i in range(6):
            for j in range(6):

                if [i,j]==self.finalState[0] or [i,j]==self.finalState[1]:
                    continue

                self.Actions[i][j] = [0,1,2,3]


    def PolicyEvaluation(self): #对当前策略进行评估
        
        while True:
            maxerror = 0#记录本次迭代最大误差

            old_values = np.copy(self.Value) #记录刷上一次的值
            #遍历每个状态，更新值
            for i in range(6):
                for j in range(6):

                    if(i==0 and j==1) or (i==5 and j==5): #跳过终止状态
                        continue
                    
                    action = self.Actions[i][j] #当前策略
                    lenth = len(action)
                    self.Value[i][j] = 0
                    for k in range(lenth):
                        ii = max(0,min(i+self.BasicAction[action[k]][0],5)) #每个动作下能到达的下一个状态
                        jj = max(0,min(j+self.BasicAction[action[k]][1],5))
                        self.Value[i][j] += 1./lenth*(self.reward+self.discntFactor*old_values[ii][jj]) 

                    maxerror = max(maxerror, abs(old_values[i][j]-self.Value[i][j]))
            

   
            self.changeserror.append(maxerror)

            if maxerror<self.error:
                break
    
    def PolicyImprove(self): #进行策略更新
        policy_state = True

        for i in range(6):
            for j in range(6):

                if [i,j]==self.finalState[0] or [i,j]==self.finalState[1]:
                    continue

                old_action = self.Actions[i][j]
                action_val = []
                for a in range(4):
                    ii = max(0,min(i+self.BasicAction[a][0],5))
                    jj = max(0,min(j+self.BasicAction[a][1],5))
                    action_val.append(self.reward+self.discntFactor*self.Value[ii][jj])

                max_action_val = max(action_val)
                current_action = [k for k in range(4) if action_val[k] == max_action_val]
                self.Actions[i][j] = current_action
                if current_action != old_action:
                    policy_state = False

        return policy_state

    #状态为0-35的一个数
    def findPolicy(self, state):

        if state==1 or state==35:
            return ["stop"]

        i = state/6 
        j = state - i*6
        current_state = [i,j]
        action = self.Actions[i][j]
        action_list = []
        
        while current_state != self.finalState[0] and current_state != self.finalState[0]:
            current_state = current_state + self.BasicAction[action[0]]
            action_list.append(self.action_convert[action[0]])

        return action_list



#值迭代的类
#成员变量：每个状态的值，每个状态的action，值迭代的终点条件和discount factor
#成员函数：FindValue: 找到收敛价值， findPolicy：找到某个状态的动作
class ValueIteration(object):
    def __init__(self,discount):

        #初始化
        self.BasicAction = [[0,1],[1,0],[0,-1],[-1,0]]  #四个基本的动作
        self.action_convert = ["right","down","left","up"]
        self.Value = np.zeros([6,6]) #所有状态的初始价值全部为0
        self.Actions = np.empty([6,6],dtype = object)#最后记录最优的策略
        self.error = 0.01
        self.discntFactor = discount
        self.reward = -1
        self.finalState = [[0,1],[5,5]]

        #记录误差和迭代次数
    
        self.changerrors = []


    def FindValue(self):
        while True:
            maxerror = 0
            old_values = np.copy(self.Value)
            for i in range(6):
                for j in range(6):

                    if [i,j]==self.finalState[0] or [i,j]==self.finalState[1]:
                        continue

                    action_val_max =  -inf
                    for k in range(4):
                        ii = max(0,min(i+self.BasicAction[k][0],5))
                        jj = max(0,min(j+self.BasicAction[k][1],5))
                        action_val_max = max(action_val_max,(self.reward+self.discntFactor*old_values[ii][jj]))

                    maxerror = max(maxerror,abs(action_val_max-self.Value[i][j]))
                    self.Value[i][j] = action_val_max
          
            self.changerrors.append(maxerror)
            if maxerror<self.error:
                break
        
        #更新actions

        for i in range(6):
            for j in range(6):
                if [i,j]==self.finalState[0] or [i,j]==self.finalState[1]:
                    continue

                action_val = []
                for k in range(4):
                    ii = max(0,min(i+self.BasicAction[k][0],5))
                    jj = max(0,min(j+self.BasicAction[k][1],5))
                    action_val.append(self.reward+self.discntFactor*self.Value[ii][jj])

                max_action_val = max(action_val)
                current_action = [k for k in range(4) if action_val[k] == max_action_val]
                self.Actions[i][j] = current_action



    def findPolicy(self, state):


        if state==1 or state==35:
            return ["stop"]
        i = state/6 
        j = state - i*6
        current_state = [i,j]
        action = self.Actions[i][j]
        action_list = []
        
        while current_state != self.finalState[0] and current_state != self.finalState[0]:
            current_state = current_state + self.BasicAction[action[0]]
            action_list.append(self.action_convert[action[0]])

        return action_list


#画出最后的结果
def plotResult(actions, values, method):
    x = [1,2,3,4,5,6]
    y = [6,6,6,6,6,6] 
    
    plt.figure(figsize=(20,8),dpi=90)
    plt.subplot(121)
    plt.plot(x,y)
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.grid(color = 'black',linewidth=1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    for i in range(6):
        for j in range(6):
            plt.text(j+0.4,5.4-i,str(values[i][j]))
    
    plt.subplot(122)
    plt.plot(x,y)
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.grid(color = 'black',linewidth=1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    direction = [[0.45,0],[0,-0.45],[-0.45,0],[0,0.45]]
    for i in range(6):
        for j in range(6):
            if actions[i][j] is None:
                plt.text(j+0.2,5.4-i,"terminal")
                continue
            for action in actions[i][j]:
                plt.quiver(j+0.5,5.5-i,direction[action][0] ,direction[action][1],angles='xy', scale_units='xy', scale=1)

    plt.savefig(method)
    plt.close()

  


def plotPerformanceCompare(PolicyChangeErrrors, ValueChangeErrors):
    len1 = len(PolicyChangeErrrors)
    len2 = len(ValueChangeErrors)

    plt.figure(figsize=(16,8),dpi=90)
    plt.subplot(121)
    plt.plot(list(range(len1)), PolicyChangeErrrors)
    plt.xlabel("iteration times")
    plt.ylabel("error changes")
    plt.title("Policy Iteration")
   

    plt.subplot(122)
    plt.plot(list(range(len2)), ValueChangeErrors)
    plt.xlabel("iteration times")
    plt.ylabel("error changes")
    plt.title("Value Iteration")
  

    plt.savefig("imgs/cmpPerformance")
    plt.close()





def main():

    disfactor = 1


    #策略迭代
    PolicyMode = PolicyIteration(disfactor)
    
    flag = False
    t1 = time.time()
    while not flag:
        PolicyMode.PolicyEvaluation()
        print(PolicyMode.Value)
        flag = PolicyMode.PolicyImprove()
    t2 = time.time()
    PolicyTime = t2-t1
    plotResult(PolicyMode.Actions,PolicyMode.Value,"imgs/Policy_Iteration_result.png")



    #值迭代

    ValueMode = ValueIteration(disfactor)
    t1 = time.time()
    ValueMode.FindValue()
    t2 = time.time()
    ValueTime = t2-t1

    plotResult(ValueMode.Actions,ValueMode.Value,"imgs/Value_Iteration_result.png")

    #比较性能
    plotPerformanceCompare(PolicyMode.changeserror,ValueMode.changerrors)
    print("The spent time of Policy Iteration: %s" %PolicyTime)
    print("The spent time of Value Iteration: %s" %ValueTime)


if __name__ =="__main__":
    main()
                    
        

        