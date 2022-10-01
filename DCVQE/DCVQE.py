import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from position import PositionalEmbedding
import random
from DivideAndConquer import TransformerD, TransformerC


class ConquerEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)



class DCVQE(nn.Module):
    '''
    Implementation of DCVQE
    '''
    def __init__(self, input_size = 4096, reduced_size = 128, head_num = 4,\
                  max_len = 240, use_res = True, seed =1024, \
                  activate_leng = 15, groups = [8,4,2,1]):
        super(DCVQE, self).__init__()
        # max_len of the video sequence
        self.max_len = max_len
        # input dim of frames
        self.input_size = input_size
        # reduced size
        self.reduced_size = reduced_size 
        # attention head nums
        self.head_num = head_num
        # whether to use residual
        self.use_res = use_res
        # random seed
        self.seed = seed
        # dims in each head 
        self.att_embedding_size = reduced_size // head_num
        # length of local mask
        self.activate_leng = int(activate_leng)
        # how many groups to divided in each attention layer
        # the length of groups indicates the layer numbers
        self.groups = groups

        # reduced layer
        self.ann = ANN(self.input_size, reduced_size, 1)
        self.droup_out = nn.Dropout2d(0.5)
        # positional embedding
        self.pe = PositionalEmbedding(d_model = reduced_size, max_len = max_len)
        # divide part attention
        self.att = nn.ModuleList(
                    [
                        TransformerD(in_size = reduced_size, max_leng = max_len, \
                                    groups = group, att_embedding_size = self.att_embedding_size, \
                                    head_num = head_num)  for group in self.groups
                    ]

                    )
        # conquer part attention
        self.shotatt = nn.ModuleList(
                        [
                            TransformerC(in_size = reduced_size, head_num = head_num)  \
                            for _ in self.groups
                        ]
                        )
        self.LayerNorm = nn.LayerNorm(reduced_size)
        self.conquer = ConquerEmbedding(2, reduced_size) 
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.conquer_fc = nn.Linear(reduced_size, 1)

    def forward(self, frame_level_quality_embedding, leng):
        '''
        :Param:  frame_level_quality_embedding, frame features extracted from CNN
                 leng,  the tensor store the length of each video

        '''
        b, s, d = frame_level_quality_embedding.shape
        video_level_quality_embedding = self.conquer(torch.tensor([[1] for i in \
        range(frame_level_quality_embedding.shape[0])]).cuda()).cuda()
        
        frame_level_quality_embedding = self.ann(frame_level_quality_embedding)
        frame_level_quality_embedding = frame_level_quality_embedding + \
        self.pe(frame_level_quality_embedding)
        
        frame_level_quality_embedding = self.LayerNorm(frame_level_quality_embedding)

        for i, group in enumerate(self.groups):
            ## DCTr
            mask_local = local_mask_divide(b, s, group, \
                          activate_time_length = self.activate_leng).cuda()
            mask_conquer = global_mask_conquer(leng, s, group).cuda()
            use_relu = True
            if i == len(self.groups) - 1:
                use_relu = False
            frame_level_quality_embedding, shot_level_quality_embedding = \
            self.att[i](frame_level_quality_embedding, video_level_quality_embedding, \
            mask_local, mask_conquer, use_relu = use_relu)
           
            if i == len(self.groups) - 1:
                video_level_quality_embedding = self.avg_pool(\
                shot_level_quality_embedding.permute(0,2,1)).permute(0,2,1)
            else:
                video_level_quality_embedding = self.shotatt[i](shot_level_quality_embedding, use_relu)
                video_level_quality_embedding = self.avg_pool(video_level_quality_embedding.permute(0,2,1)).permute(0,2,1)
        return self.conquer_fc(video_level_quality_embedding.squeeze(1))


        

class ANN(nn.Module):
    def __init__(self, input_size, reduced_size, n_ANNlayers = 1, dropout = 0.5):
        '''
        Dimension reduction

        '''
        super(ANN,self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(reduced_size, reduced_size)
    def forward(self, input):
        input = self.fc0(input)
        for i in range(self.n_ANNlayers-1):
            input = self.fc(self.dropout(F.relu(input)))
        return input




def local_mask_divide(batch_size, max_seq, groups,  activate_time_length = 6):
    group_sequence = max_seq // groups
    activate_time_length  = activate_time_length // 2
    pos = torch.triu(torch.ones(group_sequence,group_sequence), diagonal = (activate_time_length) *1 ).eq(1)
    neg = torch.triu(torch.ones(group_sequence,group_sequence), diagonal = (activate_time_length) *-1 ).eq(0)
    mask = pos+neg
    mask = mask.unsqueeze(0).expand(batch_size, groups, -1, -1)
    return mask

def global_mask_conquer(leng_seq, max_seq, groups):
    mask = []
    group_sequence = max_seq // groups
    for ls in leng_seq:
        tmp = torch.zeros(groups, group_sequence)
        leng = int(ls.item())
        conquer_tmp = torch.zeros(groups)
        for i in range(groups):
            upper_bound = max( 0, int ( leng - (group_sequence * i) ) )
            tmp[i][0:upper_bound] = 1
            conquer_tmp[i] = 1 if upper_bound != 0 else 0
        mask.append(tmp)
    mask = torch.stack(mask, dim = 0)
    mask = mask.eq(0)
    return mask 



if __name__ == '__main__':
    pass
