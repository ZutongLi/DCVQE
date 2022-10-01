import torch
import numpy as np

class RankingLoss():
    def __init__(self, img_size, level, margin = 0.25):
        '''
        img_size,  the size of input images, (the number of case)
        level,  how many level of rank. 
        '''
        self.img_size = img_size
        self.level = level
        self.margin = margin
        matMatrix = np.zeros([level, level-1])
        for i in range(level):
            dial_start_index = i
            dial_start_value = level-(i+1)
            for j in range(level-1):
                if j < dial_start_index:
                    matMatrix[i][j] = -1
                elif j == dial_start_index:
                    matMatrix[i][j] = dial_start_value
                else:
                    matMatrix[i][j] = 0
        self.matMatrix = torch.from_numpy(matMatrix).float().cuda()
    def process(self, prediction):
        prediction = prediction.view(-1, self.level)
        loss = prediction.matmul(self.matMatrix)
        loss = self.margin - loss
        loss = torch.clamp(loss, min=0)
        return loss.mean()


class CorrelationLoss():
    '''
    L_c  in DCVQE paper
    '''
    def __init__(self, margin):
        self.margin = margin
        
    def process(self, label, prediction):
        label = label.squeeze()
        prediction = prediction.squeeze()
        loss = 0
        n = label.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                l = 0 - ( (label[i]-label[j]) * (prediction[i] - prediction[j]) )
                l = torch.clamp(l, min = 0)
                loss+=l
        return loss
    def processCon(self, label, prediction):
        label = label.squeeze()
        prediction = prediction.squeeze()
        loss = 0
        n = label.shape[0]
        for i in range(1,n):
            pos = n-i
            tmp_label = torch.cat([ label[pos:], label[:pos] ])
            tmp_prediction = torch.cat([prediction[pos:], prediction[:pos]])
            l = (label - tmp_label) * (prediction - tmp_prediction)
            l = 0-l
            l = torch.clamp(l, min = 0)
            loss += l.mean()
        return loss

    def processCon2(self, label, prediction):
        label = label.squeeze()
        prediction = prediction.squeeze()
        loss = 0
        n = label.shape[0]
        for i in range(n):
            label_pivot = label[i].item()
            prediction_pivot = prediction[i].item()
            l = (label - label_pivot) * (prediction - prediction_pivot)
            l = self.margin-l
            l = torch.clamp(l, min = 0)
            loss += l.mean()
        return loss

if __name__ == '__main__':
    import time
    rl = RankingLoss(5,6)
    rl = RankingLoss(4,2)
    a = torch.rand(8)
    a = torch.Tensor([0.6,0.6,0.7,0.6,0.9,0.8,0.999,0.9]).cuda()
    print(rl.process(a))

    print('===============')
    rl2 = RankingLoss2()
    a = torch.Tensor([1,2,3,4,5,6])
    b = torch.Tensor([1,2,4,3,2,1])
    a = torch.rand([1,1024])
    b = torch.rand([1,1024])
    t1 = time.time()
    print(rl2.processCon(a,b))
    t2 = time.time()
    print(rl2.process(a,b))
    t3 = time.time()
    print(t2-t1, t3-t2)
