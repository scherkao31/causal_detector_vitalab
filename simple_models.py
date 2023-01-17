import torch
import torch.nn as nn
import random
import numpy as np



class MLP_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1):
        super(MLP_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_agents): #x contains the trajectories of the ego agent, and the considered agent
        x = self.fc1(x) 
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    
    
class RN_Classifier(nn.Module):
    def __init__(self, input_size, mlp_hidden, output_size = 1):
        super(RN_Classifier, self).__init__()
        
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
            #nn.Dropout(),
            nn.Linear(mlp_hidden, output_size),
            nn.Sigmoid()
        )


    def forward(self, x, x_agents):
        a_in = x[:, :40] #trajectory of the ego agent
        b_in = x[:, 40:] #trajectory of the considered agent
        a = self.g(a_in)
        b = self.g(b_in)
        x = self.f(a + b)
        return x
    
    

class LSTM_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1):
        super(LSTM_Classifier, self).__init__()
        
        self.lstm = nn.LSTMCell(2, hidden_size)
        
        self.hidden_size = hidden_size
        
        self.hidden2pos = nn.Linear(self.hidden_size, 2)
        
        self.fc1 = nn.Linear(hidden_size*2*20, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def init_hidden(self, batch):
        return (torch.randn(batch*2, self.hidden_size), torch.randn(batch*2, self.hidden_size))

    def forward(self, x, x_agents, training_step = 2):
        batch = x.shape[0]
        
        (h0, c0) = self.init_hidden(batch)
        
        h_states = []
        pred_pos = []
        
        x = x.reshape(batch*2, 20, 2)
        
        for input_ in x.chunk(20, dim = 1):
            input_ = input_.squeeze(1)
            h0, c0 = self.lstm(input_, (h0, c0))
            if training_step == 1:
                pos = self.hidden2pos(h0)
                pred_pos += pos.unsqueeze(0) 
            else :
                h_states += h0.unsqueeze(0)
                
        if training_step == 1:
            return  torch.stack(pred_pos).transpose(0, 1).reshape(batch, -1)
        
        else:
            states = torch.stack(h_states).transpose(0, 1).reshape(batch, 2, 20, self.hidden_size) #hidden states of every lstm cells

            states = states.reshape(batch, 2*20*self.hidden_size)

            out = self.fc1(states)
            out = self.relu(out)
            out = self.fc2(out)
            return self.sigmoid(out)
        
        
        
        