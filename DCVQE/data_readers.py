import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import copy


class DataReader(Dataset):
    def __init__(self, video_paths,  max_len=240, feat_dim=4096, scale=1):
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.paths, self.scores = self.__init_helper(video_paths)
        self.scale = scale*1.0
        print('[ DEBUG ] :: __init__ the length of paths score \
                and fps {} {}'.format(len(self.paths), len(self.scores)))

    def __init_helper(self, path):
        features, scores = [], []
        with open(path, 'r') as handle:
            for line in handle:
                line = line.strip().split('\t')
                features.append(line[0])
                scores.append(line[1])
        return features, scores
    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        score = np.load(self.scores[idx]) / self.scale
        feature = np.zeros((self.max_len, self.feat_dim))
        raw_feature = np.load(self.paths[idx])
        leng = raw_feature.shape[0]
        start_index, end_index = 0, leng
        flag = True
        concated_array = []
        while start_index < self.max_len:
            if flag:
                concated_array.append(copy.deepcopy(raw_feature))
            else:
                concated_array.append(copy.deepcopy(raw_feature)[::-1])
            start_index = end_index
            end_index += leng
            flag = not flag
        concated_array = np.concatenate(concated_array,axis = 0)
        feature = concated_array[:self.max_len]
        return feature, self.max_len, score



if __name__ == '__main__':
    data_reader = FeaturesMapReader('../Layer4/CNN_features_batched.txt', scale = 4.64)
    dataloader = DataLoader(data_reader, batch_size=2, shuffle=True, num_workers=15)
    for i, (feature, length, score) in enumerate(dataloader):
        print(i, feature.shape, length, score)



