# coding: utf-8

import torch
import torch.nn as nn
import numpy as np


''' Image Discriminator '''
class Discriminator_I(nn.Module):
    def __init__(self, k_frame=2):
        super(Discriminator_I, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(30,1024,1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024,512,1),
            nn.LeakyReLU(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(522,1024,1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024,128,1),
            nn.LeakyReLU(0.2),
        )

        self.linear1 = nn.Linear(30, 512)
        self.linear2 = nn.Linear(128,1)
        self.linear3 = nn.Linear(k_frame+1,1)
        self.activate = nn.Sigmoid()

    def forward(self, input,label):
        x = self.block1(input.transpose(1,2)).transpose(1,2)
        x = x+ self.linear1(input)
        output = torch.cat((x,label),-1)
        x = self.block2(output.transpose(1,2)).transpose(1,2)
        x = self.linear2(x).squeeze()
        x = self.linear3(x).squeeze()
        x = self.activate(x)
        return x


''' Video Discriminator '''
class Discriminator_V(nn.Module):
    def __init__(self):
        super(Discriminator_V, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(30,1000,1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1000,1000,1),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(1010,1024,1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024,128,1),
            nn.LeakyReLU(0.2),
        )

        self.linear1 = nn.Conv1d(128,1,3,padding=1)
        self.linear2 = nn.Linear(50,1)
        self.linear3 = nn.Linear(50,10)

        self.activate = nn.Sigmoid()
        self.activate1 = nn.Softmax()

    def forward(self, input,label):
        x = self.block1(input.transpose(1,2)).transpose(1,2)
        output = torch.cat((x,label),-1)
        x = self.block2(output.transpose(1,2))
        x = self.linear1(x).squeeze()
        o1 = self.linear2(x).squeeze()
        o1 = self.activate(o1)
#         o2 = self.linear3(x).squeeze()
#         o2 = self.activate1(o2)
        return o1


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.device = opt.device
        self.n_head = opt.n_head
        self.A = opt.A_seq.to(device=self.device)
        self.A_in = opt.A_i.to(device=self.device)

        self.w1     = nn.Conv1d(10,20,1)
        self.w2     = nn.Conv1d(20+opt.d_E,60,1)
        self.rnn    = nn.GRU(60,self.hidden_size,1)

        self.block1 = nn.Sequential(
            nn.Conv1d(self.hidden_size+10,512,1),
            nn.ReLU(),
            nn.Conv1d(512,1024,1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(1024,1024,1),
            nn.ReLU(),
            nn.Conv1d(1024,1024,1),
            nn.ReLU()
        )
        self.linear = nn.Conv1d(1024,30,3,padding=1)
        self.linear1 = nn.Conv1d(self.hidden_size + 10, 1024, 1)

        ''' attention '''
        self.attn_seq_1 = Self_Attn_Seq(110,opt)
        self.attn_seq_2 = Self_Attn_Seq(30,opt)

        '''graph convolution'''
        self.gc1 = Graph_Conv(1, 32 ,opt)
        self.gc2 = Graph_Conv(32, 64,opt)
        self.gc3 = Graph_Conv(64, 64,opt)
        self.gc4 = Graph_Conv(64, 128,opt)
        self.gc5 = Graph_Conv(128, 128,opt)

        self.w_g1 = nn.Conv2d(1,64,1)
        self.w_g2 = nn.Conv2d(64,128,1)

        self.w_output = nn.Conv2d(
            128,
            1,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1))

    def forward(self, label):

        n_frames = self.opt.seq_length
        noise = torch.randn(label.size(0),self.opt.seq_length,self.opt.d_E).to(self.opt.device)

        label_single = label
        label = label.unsqueeze(1).repeat(1,n_frames,1)
        label = self.w1(label.transpose(1,2)).transpose(1,2)
        input = torch.cat((label,noise),-1)
        input = self.w2(input.transpose(1,2)).transpose(1,2).transpose(0,1)
        h0 = torch.zeros(1,label.size(0),100).to(device=self.device)
        output,hn = self.rnn(input,h0)
        output = output.transpose(0,1)
        final = []
        sum = torch.rand(label.size(0), self.hidden_size).to(self.device).type(torch.float32)
        for i in range(n_frames):
            sum = sum+output[:,i,:]
            x = torch.cat((sum,label_single),1)
            final.append(x)
        final = torch.stack(final).permute(1,0,2)

        final,attn = self.attn_seq_1(final)

        x1 = self.linear1(final.transpose(1,2))+self.block1(final.transpose(1,2))

        x2 = x1 + self.block2(x1)
        output = self.linear(x2).transpose(1,2)

        output,attn = self.attn_seq_2(output)
        A_norm = self.norm(attn)

        ''' gcn '''
        A_norm = A_norm.repeat(2,1,1)
        bs, len, h = output.size()
        output = output.contiguous().view(bs * 2, len, h // 2).unsqueeze(1)

        # resident connection
        o1 = self.gc1(output,A_norm)
        o2 = self.gc2(o1,A_norm)
        o3 = self.gc3(o2,A_norm) + self.w_g1(output)
        o4 = self.gc4(o3,A_norm)
        o5 = self.gc5(o4,A_norm) + self.w_g2(o3)

        output = self.w_output(o5).squeeze()
        output = output.view(bs,len,h)
        return output

    def norm(self,attn):
        topk,indices = torch.topk(attn,self.opt.seq_connect,2)
        res = torch.zeros(attn.size()).to(attn.device)
        res = res.scatter(2,indices,topk)

        attn_tile = self.tile(res,1,15)
        attn_tile = self.tile(attn_tile,2,15)

        A_self = torch.eye(self.opt.total_joint,self.opt.total_joint).repeat(attn.size()[0],1,1).to(attn.device)

        A_hat = attn_tile*(self.A_in+self.A)+0.1*A_self+0.1*self.A_in

        D = torch.diag_embed(torch.sum(A_hat,1))
        A_norm = torch.bmm(torch.inverse(D),A_hat)
        return A_norm

    def tile(self,a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)


class Graph_Conv(nn.Module):
    def __init__(self,in_dim,hidden_dim,opt):
        super(Graph_Conv,self).__init__()

        self.hidden_dim = hidden_dim
        self.w = nn.Linear(opt.total_joint,opt.total_joint)
        self.conv = nn.Conv2d(
            in_dim,
            self.hidden_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1))

    def forward(self, input,A):
        bs, h, len, v = input.size()
        output = self.conv(input).view(bs,self.hidden_dim,-1)
        output = torch.matmul(output,A)
        output = self.w(output)
        output= output.view(bs,self.hidden_dim,len,v)

        return output




class Self_Attn_Img(nn.Module):
    def __init__(self):
        super(Self_Attn_Img, self).__init__()
        '''img attention'''
        self.hidden_dim = 55
        self.img_q = nn.Linear(1, self.hidden_dim)
        self.img_k = nn.Linear(1, self.hidden_dim)
        self.img_v = nn.Linear(1, self.hidden_dim)
        self.softmax_img = nn.Softmax(dim=2)
        self.gamma_img = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(self.hidden_dim,1)

    def forward(self, q):
        bs, len, h = q.size()
        q = q.contiguous().view(bs*2,len,h//2)
        q = q.contiguous().view(-1, h//2).unsqueeze(-1)

        residual = q
        query = self.img_q(q)
        key = self.img_k(q).transpose(1, 2)
        value = self.img_v(q)
        attn = torch.bmm(query, key)
        attn = self.softmax_img(attn)
        out = torch.bmm(attn, value)
        out = self.out(out)
        out = self.gamma_img * out + residual
        out = out.unsqueeze(-1).view(bs, len, h)
        return out


class Self_Attn_Seq(nn.Module):
    def __init__(self,in_dim,opt):
        super(Self_Attn_Seq,self).__init__()
        input_dim = in_dim
        self.chanel_in = in_dim
        self.opt = opt
        self.n_head = opt.n_head
        self.hidden_size_attention = input_dim // self.n_head
        self.w_q = nn.Linear(input_dim, self.n_head * self.hidden_size_attention)
        self.w_k = nn.Linear(input_dim, self.n_head * self.hidden_size_attention)
        self.w_v = nn.Linear(input_dim, self.n_head * self.hidden_size_attention)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (input_dim + self.hidden_size_attention)))
        nn.init.normal_(self.w_k.weight, mean=0,
                        std=np.sqrt(2.0 / (input_dim + self.hidden_size_attention)))
        nn.init.normal_(self.w_v.weight, mean=0,
                        std=np.sqrt(2.0 / (input_dim + self.hidden_size_attention)))
        self.temperature = np.power(self.hidden_size_attention, 0.5)

        self.softmax = nn.Softmax(dim=2)
        self.linear2 = nn.Linear(self.n_head * self.hidden_size_attention, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, q):
        n_head = self.n_head
        residual = q
        k, v = q, q
        bs, len, _ = q.size()
        q = self.w_q(q).view(bs, len, n_head, self.hidden_size_attention)
        k = self.w_k(k).view(bs, len, n_head, self.hidden_size_attention)
        v = self.w_v(v).view(bs, len, n_head, self.hidden_size_attention)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len, self.hidden_size_attention)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len, self.hidden_size_attention)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len, self.hidden_size_attention)

        # generate mask
        subsequent_mask = torch.triu(
            torch.ones((len, len), device=q.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(bs, -1, -1).gt(0)
        mask = subsequent_mask.repeat(n_head, 1, 1)

        # self attention
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn, v)
        output = output.view(n_head, bs, len, self.hidden_size_attention)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, len, -1)
        output = self.gamma * self.linear2(output) + residual


        attn = attn.view(n_head,bs,self.opt.seq_length,self.opt.seq_length)
        attn_avg = torch.mean(attn,0)
        return output, attn_avg





