import math
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

actionR = list([-1,0,1,0]) # up, left, down, right
actionC = list([0,-1,0,1])
iteration_time:int = 100
gamma:float = 0.9
epsilon:float = 0.3
learning_rate:float = 0.1
direct_dict:dict = {0:'up', 1:'left', 2:'down', 3:'right'}
np.set_printoptions(precision=4)
batch_size = 1

grid = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1.0]])

def Table_init(grid:np.ndarray) -> np.ndarray:
    Q_table = np.array([[0.0]* 4] * len(grid[0]) * len(grid))
    # mapping: grid[r][c] = table[r*lenC + c], lenC always 4 because 4 directions.
    # print(Q_table)
    return Q_table

def is_terminated1(r:int, c:int) -> bool:
    if r == 2 and c == 2:
        return True
    else:
        return False
    
def Q_iteration(Q_table:np.ndarray) -> None:
    global grid
    lenR = len(grid)
    lenC = len(grid[0])
    r = 0
    c = 0 # starting point
    while (1):
        direction:int = 0
        if random.random() < epsilon: # do random
            while(1): # if the move is out of bound, redo the random.
                direction = random.randint(1, 3)
                r1 = r + actionR[direction]
                c1 = c + actionC[direction]
                if 0 <= r1 < lenR and 0 <= c1 < lenC:
                    break
        else: # find max
            max_value = -1
            r_max = r
            c_max = c
            for i in range(4):
                r1:int = r + actionR[i]
                c1:int = c + actionC[i]
                if 0 <= r1 < lenR and 0 <= c1 < lenC and max_value < Q_table[r1 * lenC + c1][i]:
                    r_max = r1
                    c_max = c1
                    direction = i
                    
            r1 = r_max
            c1 = c_max
            
        if is_terminated1(r1,c1): # r1,c1 is in terminal state
            Q_table[r * lenC + c][direction] += learning_rate * (grid[r1][c1] + 0 - 
                                                              Q_table[r * lenC + c][direction])
            return None
        else: # r1,c1 is not in terminat state
            Q_table[r * lenC + c][direction] += learning_rate * (grid[r1][c1] + gamma * max(Q_table[r1 * lenC + c1]) - 
                                                              Q_table[r * lenC + c][direction])
            r = r1
            c = c1
    return None

def Trace_back(Q_table:np.ndarray) -> None:
    lenC = len(grid[0])
    r:int = 0
    c:int = 0
    while (is_terminated1(r, c) is False):
        max_index = np.argmax(Q_table[r * lenC + c])
        print(f'r = {r}\nc = {c}\ndirection = {direct_dict[max_index]}') # 0:up, 1:left, 2:down, 3:right
        # max_value = Q_table[max_index]
        r += actionR[max_index]
        c += actionC[max_index]
    # print(f'r = {r}\nc = {c}\n')
    return None

def Solution1() -> None:
    Q_table:np.ndarray = Table_init(grid)
    i = 0
    rwd_lst = [[],[]]
    flag = 0
    while (1):
        old_Q = np.copy(Q_table)
        Q_iteration(Q_table)
        # print(Q_table)
        # print('----end of an iteration-----')
        i += 1
        if i == 1000:
            break


        if np.all(np.abs(Q_table - old_Q) <= 10e-12) and flag == 0:
            print(f'converge after {i} iterations.')
            flag = 1
            # break
        rwd_lst[0].append(Q_table[0][2])
        rwd_lst[1].append(Q_table[0][3])
        # if abs(np.average(Q_table) - np.average(rwd_lst[-9:]) <= 10e-9):
        #     print(f'converge after {i} iterations.')
        #     break
        

    # x = [i for i in range(len(rwd_lst[0]))]
    # plt.scatter(x, rwd_lst[0], label='Action Down', color='blue')
    # plt.scatter(x, rwd_lst[1], label='Action Right', color='red')

    # plt.xlabel('Iteration Times')
    # plt.ylabel('Q Value of Action')
    # plt.title('Scatter Plot with Two Actions')
    # plt.legend()
    # plt.show()

    print(Q_table)
    print('----end of all iteration-----')
    print(grid)
    print('---the grid---')
    Trace_back(Q_table)
    return None

Solution1()
# Solution2()
# exit()
# Q_iteration(grid_input=copy.deepcopy(grid0))