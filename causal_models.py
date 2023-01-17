import torch
import torch.nn as nn
import random
import numpy as np


class Causal_RN_Classifier(nn.Module):
    def __init__(self, input_size, mlp_hidden, output_size):
        super(Causal_RN_Classifier, self).__init__()
        
        self.g = nn.Sequential(
            nn.Linear(input_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        
        
        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, output_size),
            nn.Sigmoid()
        )
        
        self.input_size = input_size


    def forward(self, x, x_agents):
        batch = x_agents.shape[0]
        factual = x_agents[:, 0].reshape(batch, -1) #factual trajectory
        counterfactual = x_agents[:, 1].reshape(batch, -1) #counterfactual trajectory
        a_in = factual
        b_in = counterfactual
        a = self.g(a_in)
        b = self.g(b_in)
        out = self.f(abs(a - b))
        return out
    
    


    
class Causal_LSTM_Classifier(nn.Module):
    def __init__(self, num_agents, hidden_size, mlp_hidden, output_size):
        super(Causal_LSTM_Classifier, self).__init__()
        
        self.num_agents = num_agents
        
        self.lstm = nn.LSTMCell(2, hidden_size)
        
        self.hidden_size = hidden_size

        self.sigmoid = nn.Sigmoid()
        
        self.g = nn.Sequential(
            nn.Linear(self.num_agents*20*self.hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        
        self.hidden2pos = nn.Linear(self.hidden_size, 2)
        
        self.f = nn.Sequential(
            nn.Linear(mlp_hidden + 80, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(mlp_hidden, output_size),
            nn.Sigmoid()
        )
        
    def init_hidden(self, batch):
        return (torch.randn(batch*2*self.num_agents, self.hidden_size), torch.randn(batch*2*self.num_agents, self.hidden_size))

    def forward(self, x, x_agents, training_step = 2):
        batch = x.shape[0]
        
        (h0, c0) = self.init_hidden(batch)
        h_states = []
        pred_pos = []
        
        x_agents = x_agents.reshape(batch*2*self.num_agents, 20, 2)
        
        for input_ in x_agents.chunk(20, dim = 1):
            input_ = input_.squeeze(1)
            h0, c0 = self.lstm(input_, (h0, c0))
            if training_step == 1:
                pos = self.hidden2pos(h0)
                pred_pos += pos.unsqueeze(0)
            else:
                h_states += h0.unsqueeze(0)
                
        if training_step == 1:
            return  torch.stack(pred_pos).transpose(0, 1).reshape(batch, 2, self.num_agents, 20, 2)
        else:
            states = torch.stack(h_states).transpose(0, 1).reshape(batch, 2, self.num_agents, 20, self.hidden_size)
            a = self.g(states[:, 0, :, :].reshape(batch, self.num_agents*20*self.hidden_size)) #using states of the factual scene
            b = self.g(states[:, 1, :, :].reshape(batch, self.num_agents*20*self.hidden_size)) #using states of the counterfactual scene
            c = a - b

            d = torch.cat((x, c), dim = 1)

            c = self.f(d)

            return c