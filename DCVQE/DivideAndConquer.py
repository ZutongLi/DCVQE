import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerC(nn.Module):
    '''
    Shot merger module, play a conquer part to merge all shots infomations(CM in paper)
    '''
    def __init__(self, in_size, head_num):
        super(TransformerC, self).__init__()

        if in_size % head_num != 0:
            raise Exception("[ ERROR ] TransformerC :: __init__" + \
                            "in_size {} % head_num {} != 0".format(in_size, head_num))
        self.att_embedding_size = in_size // head_num

        self.W_Query = nn.Parameter(torch.Tensor(
                    in_size, in_size,
                    ) )
        self.W_Key = nn.Parameter(torch.Tensor(
                    in_size, in_size,
                    ) )
        self.W_Value = nn.Parameter(torch.Tensor(
                    in_size, in_size,
                    ) )
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, inputs, mask = None, use_relu = True):
        keys = torch.tensordot(inputs, self.W_Key, dims = ([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims = ([-1], [0]))
        query = torch.tensordot(inputs, self.W_Query, dims = ([-1],[0]))

        querys = torch.stack(torch.split(query, \
                        self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, \
                        self.att_embedding_size, dim = 2))
        values = torch.stack(torch.split(values, \
                        self.att_embedding_size, dim = 2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)
        normalized_att_scores = F.softmax(inner_product, dim = -1)

        result = torch.matmul(normalized_att_scores, values)
        result = torch.cat(torch.split(result, 1,), dim=-1)
        result = torch.squeeze(result, dim=0)

        if use_relu:
            result = F.relu(result)
        return result

        

class TransformerD(nn.Module):
    '''
    Divide attention module, divide video sequence into shots, and do attention with local mask

    '''
    def __init__(self, in_size, max_leng, groups, att_embedding_size=8, head_num=3, use_res = True, seed = 1024):

        super(TransformerD, self).__init__()
        self.att_embedding_size = att_embedding_size
        self.groups = groups
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        if max_leng % groups != 0:
            raise Exception("[ ERROR ] DivideAndConquer :: __init__ " + \
                     "fail, max_leng {} % groups {} != 0".format(max_leng, groups))
        if in_size != head_num * att_embedding_size:
            print('[ WARNING ] DivideAndConquer :: __init__ ' +\
                 'the suggest head_num and att_embedding_size should satisfied the ' + \
                 'formular att_embedding_size * head_num = in_size')
        self.LayerNorm = nn.LayerNorm(in_size)
        
        self.W_Query = nn.Parameter(torch.Tensor(
                        in_size, self.att_embedding_size * self.head_num,
                        ) )

        self.W_Key = nn.Parameter(torch.Tensor(
                        in_size, self.att_embedding_size * self.head_num,
                        ) )
        self.W_Value = nn.Parameter(torch.Tensor(
                        in_size, self.att_embedding_size * self.head_num,
                        ) )
        self.W_Conquer = nn.Parameter(torch.Tensor(
                        in_size, self.att_embedding_size * self.head_num,
                        ) )
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        if self.use_res:
            self.W_res = nn.Parameter(torch.Tensor(
                            in_size, self.att_embedding_size * self.head_num,
                        ) )
            self.WC_res = nn.Parameter(torch.Tensor(
                            in_size, self.att_embedding_size * self.head_num,
                        ) )
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)


    def forward(self, inputs, conquer, mask_local = None, mask_conquer = None, use_relu = True):
        b, s, d = inputs.shape
        groups_inputs = inputs.view([b, self.groups, s//self.groups, d]).\
                            view(b*self.groups, s//self.groups, d)
        groups_conquer = conquer.expand(b,self.groups,d).unsqueeze(2).\
                          contiguous().view(b*self.groups,1,d)
        ## group, b, sequence, dim
        #groups_inputs = groups_inputs.view(b*self.groups, s//self.groups, d)
        ## group, batch, sequence
        mc_b, mc_g, mc_s = mask_conquer.shape
        mask_conquer = mask_conquer.view(mc_b*mc_g, mc_s)
        ml_b, ml_g, ml_s, ml_d = mask_local.shape
        mask_local = mask_local.view(ml_b*ml_g, ml_s, ml_d)


        results, conquers = self.attention_divide(groups_inputs, \
                            groups_conquer, mask_local, mask_conquer, use_relu)
        results = results.view(b, s, d)
        conquers = conquers.view(b, self.groups, d)
        return results, conquers 

    def attention_divide(self, inputs, conquer, mask_local = None, mask_conquer = None, use_relu = True):
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_Key, dims = ([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims = ([-1], [0]))
        query_conquer = torch.tensordot(conquer, self.W_Query, dims = ([-1],[0]))

        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim = 2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim = 2)) 
        query_conquer = torch.stack(torch.split(query_conquer, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)
        conquer_product = torch.einsum('bnik,bnjk->bnij', query_conquer, keys).squeeze(2)
        if mask_local is not None:
            mask = mask_local.repeat(self.head_num, 1,1,1)
            inner_product = inner_product.masked_fill(mask, -np.inf)
            normalized_att_scores = F.softmax(inner_product, dim = -1)
        else:
            normalized_att_scores = F.softmax(inner_product, dim = -1)
        if mask_conquer is not None:
            mask = mask_conquer.repeat(self.head_num ,1,1)
            conquer_product = conquer_product.masked_fill(mask, -np.inf)
            conquer_att_scores = F.softmax(conquer_product, dim = -1)
        else:
            conquer_att_scores = F.softmax(conquer_product, dim = -1)
    
        result = torch.matmul(normalized_att_scores, values)
        result = torch.cat(torch.split(result, 1,), dim=-1)
        result = torch.squeeze(result, dim=0)
        result = F.relu(result)
        result = self.LayerNorm(result)
        
        conquer_result = torch.matmul(conquer_att_scores.unsqueeze(2), values)
        conquer_result = torch.cat(torch.split(conquer_result, 1,), dim=-1)
        conquer_result = torch.squeeze(conquer_result, dim=0)
        if use_relu:
            conquer_result = F.relu(conquer_result)
        
        return result, conquer_result
    
    def attention_conquer(self, inputs, conquer, mask_conquer):
        query_conquer = torch.tensordot(conquer, self.W_Conquer, dims = ([-1],[0]))
        query_conquer = torch.stack(torch.split(query_conquer, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(inputs, self.att_embedding_size, dim = 2))
        conquer_product = torch.einsum('bnik,bnjk->bnij', query_conquer, keys).squeeze(2)
        if mask_conquer is not None:
            mask = mask_conquer.repeat(self.head_num ,1,1)
            conquer_product = conquer_product.masked_fill(mask, -np.inf)
            conquer_att_scores = F.softmax(conquer_product, dim = -1)
        conquer_result = torch.matmul(conquer_att_scores.unsqueeze(2), keys)
        conquer_result = torch.cat(torch.split(conquer_result, 1,), dim=-1)
        conquer_result = torch.squeeze(conquer_result, dim=0)
        conquer_result = F.relu(conquer_result)
        return conquer_result
       

 

if __name__ == '__main__':
    pass
