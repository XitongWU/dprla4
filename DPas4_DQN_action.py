import math
import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque


class Action_Network(nn.Module):
    def __init__(self) -> None:
        super(Action_Network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3, 12)
        self.linear2 = nn.Linear(12, 4)
        self.linear3 = nn.Linear(4,1)
        self.activation = nn.ReLU()
        # self.double()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x
        

actionR:list[int] = [-1, 0, 1, 0] # up, left, down, right
actionC:list[int] = [0, -1, 0, 1]
iteration_time:int = 100
gamma:float = 0.9
epsilon:float = 0.3
learning_rate:float = 0.005
direct_dict:dict = {0:'up', 1:'left', 2:'down', 3:'right'}
np.set_printoptions(precision=4)
batch_size = 1
lenR = 5
lenC = 5
exp_buffer:deque[list[int]] = deque(maxlen=100)

Qnetwork = Action_Network()
Qhat = copy.deepcopy(Qnetwork)
lossFunc = nn.MSELoss()
optimizer = optim.SGD(Qnetwork.parameters(), lr=learning_rate)

grid_reward = np.array(([0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,1]))
     
                        
def get_state(row:int, col:int, action:int) -> torch.Tensor:
    state = torch.tensor(np.array([row, col, action]), dtype=torch.float32)
    state = state.unsqueeze(0)
    # state_lst = [state] * batch_size
    return state # Tensor, 1 * 3

def init_buffer(amount:int) -> None:
    row_lst = np.random.randint(0, lenR, amount)
    col_lst = np.random.randint(0, lenC, amount)
    
    for i in range(amount): # how much size you want to init the buffer
        action, r1, c1 = explore(row_lst[i],col_lst[i])
        if row_lst[i] != lenR - 1 and col_lst[i] != lenC - 1:
            exp_buffer.append([row_lst[i], col_lst[i], action, grid_reward[r1][c1], r1, c1])
        # if it's a end state, do not add into the buffer.
    # print(exp_buffer)
    return

def explore(row:int, col:int) -> int:
    next_direction = 0
    if random.random() < epsilon: # random explore
        while (1):
            direction = random.randint(0,3)
            r1 = row + actionR[direction]
            c1 = col + actionC[direction]
            if 0 <= r1 < lenR and 0 <= c1 < lenC:
                next_direction = direction
                break
    else:
        out_max = -1
        r_max = 0
        c_max = 0
        for direction in range(4):
            r1 = row + actionR[direction]
            c1 = col + actionC[direction]
            
            if 0 <= r1 < lenR and 0 <= c1 < lenC: # Greedy, find Q(t+1) max.
                state = get_state(r1, c1, direction)
                out = Qnetwork(state)[0,0].item()
                if out > out_max:
                    out_max = out
                    r_max = r1
                    c_max = c1
                    next_direction = direction
            
        r1 = r_max
        c1 = c_max
        
    return next_direction, r1, c1

def get_batch(batch_size:int) -> torch.Tensor:
    
    batch_sequence = np.arange(len(exp_buffer))
    np.random.shuffle(batch_sequence)
    batch_sequence = batch_sequence[:batch_size]
    
    tensorQ = []
    tensorQhat = []
    for i in range(batch_size):
        
        tensorQ.append(get_state(exp_buffer[batch_sequence[i]][0],  # si row
                                 exp_buffer[batch_sequence[i]][1],  # si col
                                 exp_buffer[batch_sequence[i]][2])) # ai

        maxQ = 0
        for direction in range(4):
            input_hat = get_state(exp_buffer[batch_sequence[i]][4], # si+1 row
                                  exp_buffer[batch_sequence[i]][5],direction) # si+1 col
            out_hat = Qhat(input_hat)
            if out_hat[0,0].item() > maxQ:
                maxQ = out_hat[0,0].item()
        qhat = torch.tensor([exp_buffer[batch_sequence[i]][3] + gamma * maxQ], dtype=torch.float32) # ri
        qhat.unsqueeze(0)
        tensorQhat.append(qhat)

    tensorQ = torch.stack(tensorQ, dim=0)
    tensorQ = Qnetwork(tensorQ)
    tensorQhat = torch.stack(tensorQhat, dim=0)
    # print(tensorQ.dtype)
    # print(tensorQhat.dtype)
    loss = lossFunc(tensorQ, tensorQhat)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return # tensor with batch_size * 1 * 3

def TrainDQN() -> None:
    global batch_size, exp_buffer, Qnetwork, Qhat, iteration_time
    iteration_time = 300
    batch_size = 5
    r = 3
    c = 4
    a = [torch.Tensor] * 4
    for i in range(4):
        a[i] = get_state(r,c,i)
    
    for i in range(iteration_time):
        Qhat = copy.deepcopy(Qnetwork)
        # print(Qnetwork(a))
        if i % 10 == 0:
            for k in range(4):
                print(Qnetwork(a[k]))
            print('-----') # up, left, down, right
        init_buffer(amount=25)
        get_batch(batch_size)
        
    return None

# print(get_state(2,2))
# get_state(2,2,3)
TrainDQN()