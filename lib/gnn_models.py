import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax,add_remaining_self_loops
import math
import lib.utils as utils


class TemporalEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()
        self.d_hid = d_hid
        self.div_term = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)]) #[20]
        self.div_term = torch.reshape(self.div_term,(1,-1))
        self.div_term = nn.Parameter(self.div_term,requires_grad=False)

    def forward(self, t):
        '''

        :param t: [n,1]
        :return:
        '''
        t = t.view(-1,1)
        t = t *200  # scale from [0,1] --> [0,200], align with 'attention is all you need'
        position_term = torch.matmul(t,self.div_term)
        position_term[:,0::2] = torch.sin(position_term[:,0::2])
        position_term[:,1::2] = torch.cos(position_term[:,1::2])

        return position_term



class GTrans(MessagePassing):

    def __init__(self, n_heads=2,d_input=6, d_k=6,dropout = 0.1,**kwargs):
        super(GTrans, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_k//n_heads
        self.d_q = d_k//n_heads
        self.d_e = d_k//n_heads
        self.d_sqrt = math.sqrt(d_k//n_heads)

        #Attention Layer Initialization
        self.w_k_list_same = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for i in range(self.n_heads)])
        self.w_v_list_same = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for i in range(self.n_heads)])
        self.w_v_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])

        #self.w_transfer = nn.ModuleList([nn.Linear(self.d_input*2, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_transfer = nn.ModuleList([nn.Linear(self.d_input +1, self.d_k, bias=True) for i in range(self.n_heads)])

        #initiallization
        utils.init_network_weights(self.w_k_list_same)
        utils.init_network_weights(self.w_k_list_diff)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list_same)
        utils.init_network_weights(self.w_v_list_diff)
        utils.init_network_weights(self.w_transfer)


        #Temporal Layer
        self.temporal_net = TemporalEncoding(d_input)

        #Normalization
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_value,time_nodes,edge_same):

        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value, edge_same=edge_same, residual=residual)

    def message(self, x_j,x_i,edge_index_i, edges_temporal,edge_same):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        edge_same = edge_same.view(-1,1)
        for i in range(self.n_heads):
            k_linear_same = self.w_k_list_same[i]
            k_linear_diff = self.w_k_list_diff[i]
            q_linear = self.w_q_list[i]
            v_linear_same = self.w_v_list_same[i]
            v_linear_diff = self.w_v_list_diff[i]
            w_transfer = self.w_transfer[i]

            edge_temporal_true = self.temporal_net(edges_temporal)
            edges_temporal = edges_temporal.view(-1,1)
            x_j_transfer = F.gelu(w_transfer(torch.cat((x_j, edges_temporal), dim=1))) + edge_temporal_true

            attention = self.each_head_attention(x_j_transfer,k_linear_same,k_linear_diff,q_linear,x_i,edge_same) #[4,1]
            attention = torch.div(attention,self.d_sqrt)
            attention_norm = softmax(attention,edge_index_i) #[4,1]
            sender_same = edge_same * v_linear_same(x_j_transfer)
            sender_diff = (1-edge_same) * v_linear_diff(x_j_transfer)
            sender = sender_same + sender_diff

            message  = attention_norm * sender #[4,3]
            messages.append(message)

        message_all_head  = torch.cat(messages,1)

        return message_all_head

    def each_head_attention(self,x_j_transfer,w_k_same,w_k_diff,w_q,x_i,edge_same):
        x_i = w_q(x_i) #receiver #[num_edge,d*heads]

        # wraping k

        sender_same = edge_same * w_k_same(x_j_transfer)
        sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        sender = sender_same + sender_diff #[num_edge,d]

       # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender,1),torch.unsqueeze(x_i,2))

        return torch.squeeze(attention,1)

    def update(self, aggr_out,residual):
        x_new = residual + F.gelu(aggr_out)

        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class NRIConv(nn.Module):
    """MLP decoder module."""

    def __init__(self, in_channels, out_channels, dropout=0., skip_first=False):
        super(NRIConv, self).__init__()

        self.edge_types = 2
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_channels, out_channels) for _ in range(self.edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(out_channels, out_channels) for _ in range(self.edge_types)])
        self.msg_out_shape = out_channels
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(in_channels + out_channels, out_channels)
        self.out_fc2 = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)


        #input data
        self.rel_type = None
        self.rel_rec = None
        self.rel_send = None



    def forward(self, inputs, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        '''

        :param inputs: [b,n_ball,feat]
        :param rel_type: [b,20,2]
        :param rel_rec:  [20,5] : [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
        :param rel_send: [20,5]: [1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3]
        :param pred_steps:10
        :return:
        '''
        rel_type = self.rel_type
        rel_rec = self.rel_rec
        rel_send = self.rel_send

        # Node2edge
        receivers = torch.matmul(rel_rec, inputs)  # [b,20,256], 20edges, receiver features: [20,4]
        senders = torch.matmul(rel_send, inputs)  # [b,20,256], 20edges, receiver_features: [20,4]
        pre_msg = torch.cat([senders, receivers], dim=-1)  # 【b,20,256*2]

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),self.msg_out_shape)  # [b,20,256]

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = self.dropout(msg)
            msg = F.relu(self.msg_fc2[i](msg))  # 【b,20,256]
            msg = msg * rel_type[:, :, i:i + 1] # [b,20,256]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1) #[b,5,256]

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = self.dropout(F.relu(self.out_fc1(aug_inputs)))
        pred = self.dropout(F.relu(self.out_fc2(pred)))

        # Predict position/velocity difference
        return inputs + pred


class GeneralConv(nn.Module):
    '''
    general layers
    '''
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'GTrans':
            self.base_conv = GTrans(n_heads,in_hid,out_hid,dropout)
        elif self.conv_name == "NRI":
            self.base_conv = NRIConv(in_hid,out_hid,dropout)


    def forward(self, x, edge_index, edge_time, x_time,edge_same):
        if self.conv_name == 'GTrans':
            return self.base_conv(x, edge_index, edge_time, x_time,edge_same)
        elif self.conv_name =="NRI":
            return self.base_conv(x)

class GNN(nn.Module):
    '''
    wrap up multiple layers
    '''
    def __init__(self, in_dim, n_hid,out_dim, n_heads, n_layers, dropout = 0.2, conv_name = 'GTrans',aggregate = "add"):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.drop = nn.Dropout(dropout)
        self.adapt_ws = nn.Linear(in_dim,n_hid)
        self.sequence_w = nn.Linear(n_hid,n_hid) # for encoder
        self.out_w_ode = nn.Linear(n_hid,out_dim)
        self.out_w_encoder = nn.Linear(n_hid,out_dim*2)

        #initialization
        utils.init_network_weights(self.adapt_ws)
        utils.init_network_weights(self.sequence_w)
        utils.init_network_weights(self.out_w_ode)
        utils.init_network_weights(self.out_w_encoder)

        # Normalization
        self.layer_norm = nn.LayerNorm(n_hid)
        self.aggregate = aggregate
        for l in range(n_layers):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid,  n_heads, dropout))

        if conv_name == 'GTrans':
            self.temporal_net = TemporalEncoding(n_hid)
            #self.w_transfer = nn.Linear(self.n_hid * 2, self.n_hid, bias=True)
            self.w_transfer = nn.Linear(self.n_hid + 1, self.n_hid, bias=True)
            utils.init_network_weights(self.w_transfer)

    def forward(self, x,edge_time=None, edge_index=None, x_time=None, edge_same=None,batch= None, batch_y = None):  #aggregation part
        h_0 = F.relu(self.adapt_ws(x))
        h_t = self.drop(h_0)
        h_t = self.layer_norm(h_t)

        for gc in self.gcs:
            h_t = gc(h_t, edge_index, edge_time, x_time,edge_same)  #[num_nodes,d]

        ### Output
        if batch!= None:  ## for encoder
            batch_new = self.rewrite_batch(batch,batch_y) #group by balls
            if self.aggregate == "add":
                h_ball = global_mean_pool(h_t,batch_new) #[num_ball,d], without activation

            elif self.aggregate == "attention":


                #h_t = F.gelu(self.w_transfer(torch.cat((h_t, edges_temporal), dim=1))) + edges_temporal
                x_time = x_time.view(-1,1)
                h_t = F.gelu(self.w_transfer(torch.cat((h_t, x_time), dim=1))) + self.temporal_net(x_time)
                attention_vector = F.relu(self.sequence_w(global_mean_pool(h_t,batch_new))) #[num_ball,d] ,graph vector with activation Relu
                attention_vector_expanded = self.attention_expand(attention_vector,batch,batch_y) #[num_nodes,d]
                attention_nodes = torch.sigmoid(torch.squeeze(torch.bmm(torch.unsqueeze(attention_vector_expanded,1),torch.unsqueeze(h_t,2)))).view(-1,1) #[num_nodes]
                nodes_attention = attention_nodes * h_t #[num_nodes,d]
                h_ball = global_mean_pool(nodes_attention,batch_new) #[num_ball,d] without activation
          

            h_out = self.out_w_encoder(h_ball) #[num_ball,2d]
            mean,mu = self.split_mean_mu(h_out)
            mu = mu.abs()
            return mean,mu

        else:  # for ODE
            # h_t [n_ball,d]
            h_out = self.out_w_ode(h_t)

        return h_out

    def rewrite_batch(self,batch, batch_y):
        assert (torch.sum(batch_y).item() == list(batch.size())[0])
        batch_new = torch.zeros_like(batch)
        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            batch_new[current_index:current_index + ball_time] = group_num
            group_num += 1
            current_index += ball_time.item()

        return batch_new

    def attention_expand(self,attention_ball, batch,batch_y):
        '''

        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = batch.size()[0]
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim)
        if attention_ball.device != torch.device("cpu"):
            new_attention = new_attention.cuda()

        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            new_attention[current_index:current_index+ball_time] = attention_ball[group_num]
            group_num +=1
            current_index += ball_time.item()

        return new_attention

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        res = h[:,:last_dim], h[:,last_dim:]
        return res





