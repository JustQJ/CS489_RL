


import time
import matplotlib.pyplot as plt
import numpy as np

#MC 构造一个类
'''
1 一个产生随机episodes的函数
2 一个计算first visit的MC
3 一个计算every visit的MC

'''




class MC(object):

    def __init__(self, discount):
        
        self.FirstValue =  np.random.randn(6,6)  #初始化所有位置的值
        self.EveryValue =  np.random.randn(6,6) #初始化所有位置的值
        self.FirstValue[0][1] = 0
        self.FirstValue[5][5] = 0
        self.EveryValue[0][1] = 0
        self.EveryValue[5][5] = 0

        self.reward = -1
        self.discount = discount
        self.action = [[-1,0],[0,1],[1,0],[0,-1]]
        self.terminalState = np.array([[0,1],[5,5]])


    #产生一个随机序列
    def produce_random_episodes(self):

        #记录产生的状态
        episodesList = []

        state = np.random.randint(0,6,2) #产生一个初始状态
        episodesList.append(state.copy())
        while (state!=self.terminalState[0]).any() and (state!=self.terminalState[1]).any():
            a = np.random.randint(0,4,1) #随机一个动作
            state[0] = max(0,min(5,state[0]+self.action[a[0]][0]))
            state[1] = max(0,min(5,state[1]+self.action[a[0]][1]))

            episodesList.append(state.copy())

        return episodesList

    def first_visit_MC(self,maxEpisodes):
        #定义一个记录每个状态出现次数的数组
        count = np.zeros((6,6))

        times = 1
        error = []
        while times < maxEpisodes: #决定最大的迭代次数

            tempcount = np.zeros((6,6)) #记录本次的出现次数，1次或者0次
            episd = self.produce_random_episodes() #产生一个序列

            n = len(episd)

            maxerror = 0
            for i in range(n-1): #终止状态始终是0，不需要计算

                if tempcount[episd[i][0]][episd[i][1]] == 0: #第一次出现
                    tempcount[episd[i][0]][episd[i][1]] = 1
                    count[episd[i][0]][episd[i][1]] +=1 #次数加一
                    Gt = 0
                    if self.discount == 1:
                        Gt = self.reward*(n-i-1)
                    else:
                        Gt = self.reward*((1-self.discount**(n-i-1))/(1-self.discount))
                    
                    oldval = self.FirstValue[episd[i][0]][episd[i][1]]
                    self.FirstValue[episd[i][0]][episd[i][1]] += 1./count[episd[i][0]][episd[i][1]]*(Gt-self.FirstValue[episd[i][0]][episd[i][1]])

                    maxerror = max(maxerror,abs(oldval-self.FirstValue[episd[i][0]][episd[i][1]]))
            times+=1
            error.append(maxerror)

        return error

    def every_visit_MC(self,maxEpisodes):
        #定义一个记录每个状态出现次数的数组
        count = np.zeros((6,6))

        times = 1
        error = []
        while times < maxEpisodes: #决定最大的迭代次数

            episd = self.produce_random_episodes() #产生一个序列

            n = len(episd)
            maxerror = 0
            for i in range(n-1): #终止状态始终是0，不需要计算
                    count[episd[i][0]][episd[i][1]] +=1 #次数加一
                    Gt = 0
                    if self.discount == 1:
                        Gt = self.reward*(n-i-1)
                    else:
                        Gt = self.reward*((1-self.discount**(n-i-1))/(1-self.discount))
                    oldval = self.EveryValue[episd[i][0]][episd[i][1]]
                    self.EveryValue[episd[i][0]][episd[i][1]] += 1./count[episd[i][0]][episd[i][1]]*(Gt-self.EveryValue[episd[i][0]][episd[i][1]])
                    maxerror = max(maxerror,abs(oldval-self.EveryValue[episd[i][0]][episd[i][1]]))
            times+=1
            error.append(maxerror)
        return error


class TD(object):
    def __init__(self,discount,stepSize):

        self.Value = np.random.randn(6,6) #随机初始化
      
        self.Value[0][1] = 0
        self.Value[5][5] = 0
      
        self.reward = -1
        self.discount = discount
        self.stepSize = stepSize
        self.action = [[-1,0],[0,1],[1,0],[0,-1]]
        self.terminalState = np.array([[0,1],[5,5]])
        

    def TD_find(self, maxepisods):
        
        times = 1
        error = []
        while times<maxepisods:
            self.stepSize = self.stepSize*0.99
            maxerror = 0
            state = np.random.randint(0,6,2) #随机初始状态
            while (state!=self.terminalState[0]).any() and (state!=self.terminalState[1]).any():
                a = np.random.randint(0,4,1) #随机一个动作

                oldval = self.Value[state[0]][state[1]]
                oldstate = state.copy()
                state[0] = max(0,min(5,state[0]+self.action[a[0]][0]))
                state[1] = max(0,min(5,state[1]+self.action[a[0]][1]))
                self.Value[oldstate[0]][oldstate[1]] = self.stepSize*(self.reward + self.discount*self.Value[state[0]][state[1]]) + (1-self.stepSize)*self.Value[oldstate[0]][oldstate[1]]

                maxerror = max(maxerror, abs(oldval-self.Value[oldstate[0]][oldstate[1]]))
            times+=1

            error.append(maxerror)
        return error



def plotResult(values1,values2,values3):
    x = [1,2,3,4,5,6]
    y = [6,6,6,6,6,6] 

    values1 = np.around(values1,2)
    values2 = np.around(values2,2)
    values3 = np.around(values3,2)

    plt.figure(figsize=(20,8),dpi=90)
    plt.subplot(131)
    plt.plot(x,y)
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.grid(color = 'black',linewidth=1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    for i in range(6):
        for j in range(6):
            plt.text(j+0.2,5.4-i,str(values1[i][j]))
    plt.title("MC First-Visit")

    plt.subplot(132)
    plt.plot(x,y)
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.grid(color = 'black',linewidth=1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    for i in range(6):
        for j in range(6):
            plt.text(j+0.2,5.4-i,str(values2[i][j]))

    plt.title("MC every-Visit")


    plt.subplot(133)
    plt.plot(x,y)
    plt.xlim((0,6))
    plt.ylim((0,6))
    plt.grid(color = 'black',linewidth=1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    for i in range(6):
        for j in range(6):
            plt.text(j+0.2,5.4-i,str(values3[i][j]))

    plt.title("TD")
    
    

    plt.savefig("imgs/Values.png")
    plt.close()


def poltError(error1,error2,error3):

    plt.figure(figsize=(20,8),dpi=90)
    plt.subplot(131)
    plt.plot(list(range(len(error1))),error1)
    plt.xlabel("Iteration episodes")
    plt.ylabel("Error changes")
    plt.title("MC First-Visit")


    plt.subplot(132)
    plt.plot(list(range(len(error2))),error2)
    plt.xlabel("Iteration episodes")
    plt.ylabel("Error changes")
    plt.title("MC every-Visit")



    plt.subplot(133)
    plt.plot(list(range(len(error3))),error3)
    plt.xlabel("Iteration episodes")
    plt.ylabel("Error changes")
    plt.title("TD")

    plt.savefig("imgs/Errors.png")


def main():


    #实验MC
    MC_discount = 1
    MC_maxEpsides = 5000
    MC_model = MC(MC_discount)
    t1 = time.time()
    Error1 = MC_model.first_visit_MC(MC_maxEpsides)
    t2 = time.time()
    Error2 = MC_model.every_visit_MC(MC_maxEpsides)
    t3 = time.time()

    print("The spent time of MC first visit: %s" %(t2-t1))
    print("The spent time of MC every visit: %s" %(t3-t2))
    


    #实验TD
    TD_discount = 1
    TD_maxEpsides = 5000
    TD_stepsize = 0.8
    TD_model = TD(TD_discount,TD_stepsize)
    t4 = time.time()
    Error3 = TD_model.TD_find(TD_maxEpsides)
    t5 = time.time()
    print("The spent time of TD: %s" %(t5-t4))


    #画图
    plotResult(MC_model.FirstValue,MC_model.EveryValue,TD_model.Value)

    poltError(Error1,Error2,Error3)




if __name__ == "__main__":
    main()



    




    

