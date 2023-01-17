import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np



# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


    
    
class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32),#.cuda(),
            torch.nn.InstanceNorm1d(64),#.cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, batch):
        graph_embeded_data = []
        for i in range(batch):
            curr_seq_embedding_traj = obs_traj_embedding[:, i, :, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj).unsqueeze(0)
            graph_embeded_data += [curr_seq_graph_embedding]
        graph_embeded_data = torch.cat(graph_embeded_data, dim=0)
        return graph_embeded_data


    

    
class STGAT_like_Classifier(nn.Module):
    def __init__(self, time_step_traj = 20, hidden_size = 32, mlp_hidden = 32, output_size = 1, graph_lstm_hidden_size = 32, hidden_units = "16", heads = "4,1"):
        super(STGAT_like_Classifier, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.time_step_traj = time_step_traj
        
        self.hidden_units = hidden_units
        self.heads = heads
        
        #self.num_agents = num_agents
        
        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        
        n_units = (
            [self.hidden_size]
            + [int(x) for x in self.hidden_units.strip().split(",")]
            + [self.graph_lstm_hidden_size]
        )
        n_heads = [int(x) for x in self.heads.strip().split(",")]
        
        self.lstm = nn.LSTMCell(2, hidden_size)
        self.lstm_g = nn.LSTMCell(hidden_size, hidden_size)
        
        
        
        self.hidden2pos = nn.Linear(self.hidden_size, 2)
        self.hidden2pos_g = nn.Linear(self.hidden_size*2, 2)
        
        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=0, alpha= 0.2
        )

        self.sigmoid = nn.Sigmoid()
        
        mlp_hidden = self.hidden_size
        
        self.dim_entry_g = self.hidden_size*2*20
        
        self.g = nn.Sequential(
            nn.Linear(self.dim_entry_g, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        
        
        self.f = nn.Sequential(
            nn.Linear(mlp_hidden + self.time_step_traj*2*2, mlp_hidden),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(mlp_hidden, output_size),
            nn.Sigmoid()
        )
        
        
    def init_hidden(self, batch):
        return (torch.randn(batch, self.hidden_size), torch.randn(batch, self.hidden_size))

    def forward(self, x_pair, x, training_step_ = 1):

        batch = x.shape[0]
        num_agents = x.shape[1]

        
        (h0, c0) = self.init_hidden(batch*num_agents)
        (h0_g, c0_g) = self.init_hidden(batch*num_agents)
        h_states = []
        pred_pos = []
        
        h_g_states = []
        pred_pos_g = []
        

        x = x.reshape(batch*num_agents, 20, 2)

        temp = 0
        for input_ in x.chunk(20, dim = 1):
            input_ = input_.squeeze(1)

            h0, c0 = self.lstm(input_, (h0, c0))
            if training_step_ == 1:
                if temp < 19:
                    pos = self.hidden2pos(h0)
                    pred_pos += pos.unsqueeze(0)
            else:

                h_states += h0.unsqueeze(0)
            temp += 1
            
            
        if  training_step_ != 1:
            in_graph = torch.stack(h_states).reshape(20, batch, num_agents, -1)
            graph_lstm_input = self.gatencoder(
                    in_graph, batch
                )

            state_gat = graph_lstm_input.transpose(1, 2)
            to_lstm = state_gat.reshape(batch*num_agents, 20, 32)
            
            temp = 0
            for i, input_ in enumerate(to_lstm.chunk(20, dim = 1)):
                input_ = input_.squeeze(1)
                h0_g, c0_g = self.lstm_g(input_, (h0_g, c0_g))

                to_pred = torch.cat((h_states[i], h0_g), dim = 1)

                if training_step_ == 2:
                    if temp < 19:
                        pos = self.hidden2pos_g(to_pred)
                        pred_pos_g += pos.unsqueeze(0)
                else:
                    h_g_states += to_pred.unsqueeze(0) #contains state of lstm and gat
                temp += 1
                
        if training_step_ == 1:
            out = torch.stack(pred_pos).transpose(0, 1).reshape(batch, num_agents, -1).reshape(batch, num_agents, 19, 2)
            
        if training_step_ == 2:
            out = torch.stack(pred_pos_g).transpose(0, 1).reshape(batch, num_agents, -1).reshape(batch, num_agents, 19, 2)
            
        if training_step_ == 3:
            states = torch.stack(h_g_states).transpose(0, 1).reshape(batch, num_agents, 20, -1)
            ego = states[:, 0].reshape(batch, -1)
            agent = states[:, 1].reshape(batch, -1)
            a = self.g(ego)
            b = self.g(agent)
            c = a - b
            in_f = torch.cat((x_pair, c), dim = 1)
            out = self.f(in_f)

        return out