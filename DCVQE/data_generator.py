import os
from resnet import resnet50
import h5py
import utils
import torch
import numpy as np
import torch.nn as nn
from myTransforms import cv2Norm, cv2Resize

class DataGen():
    def __init__(self,  dest_path, pretrained_weight):
        utils.check_path(dest_path)
        self.featureEx = FeatureEx(pretrained_weight = pretrained_weight).cuda().eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.dest_path = dest_path

    def process_inp_paths(self, videos, scores, out_haldle):
        
        for idx in range(len(videos)):
            video = videos[idx]
            print('[ DEBUG ] now processing {} ......'.format(video))
            name = video.split('/')[-1].split('.')[0]
            score = float(scores.get(name, '-101') )
            prn_score = score
            if score<=-100:
                print('[ ERROR ] cannot find the score for video {}'.format(name ))
                continue
            
            dest_features = os.path.join(self.dest_path, '{}_features.npy'.format(name) )
            dest_scores = os.path.join(self.dest_path, '{}_scores.npy'.format(name) )
            if os.path.exists(dest_features):
                print('[ DEBUG ] {} already gened!'.format(name))
                continue
            try:
                features, _ = self.get_features(video, frame_batch_size = 6, resize_to = 10240, want_score =False)
            except Exception as err:
                print('[ ERROR ] process_inp_paths, ', err)
                continue
            np.save(dest_features, features.cpu().numpy() )
            np.save(dest_scores, score)
            out_haldle.write("{}\t{}\n".format(dest_features, dest_scores))
            print('[ DEBUG ] shape of features {}, and scores {}'.format(features.shape, prn_score))

    def process(self, out_handle):
        '''
        convert videos to numpy data
        '''
        for idx in range(len(self.all_mp4s)):
            mp4 = self.all_mp4s[idx]
            features, score = self.get_features(mp4)
            dest_feature = os.path.join(self.dest_path,   'KoNViD_{}_resnet-50.npy'.format(idx))
            dest_score = os.path.join(self.dest_path, 'KoNViD_{}_score.npy'.format(idx))
            np.save(dest_feature, features.cpu().numpy())
            np.save(dest_score, score)
            out_handle.write(dest_feature+'\t'+dest_score + '\n')
    
    def get_features(self, video_path, frame_batch_size = 16, \
                         Blur_Features = None, want_score = True, resize_to = 1024):
        if Blur_Features is None:
            frames = utils.get_all_frames(video_path)
            
        else:
            frames = Blur_Features
        print(frames.shape)
        vid = self.__get_vid(video_path)
        if want_score:
            score = self.scores_mapper[vid]
        else:
            score = 0.0
        video_length = frames.shape[0]
        video_channel = frames.shape[3]
        video_height = frames.shape[1]
        video_width = frames.shape[2]
        print('[ DEBUG ] DataGen::get_features now processing {}, video_length {},'.format(video_path, video_length) \
               + ' video_channel {}, video_height {}, video_width {}'.format(video_channel , video_height , video_width))
        transformed_video = []
        for frame in frames:
            transformed_video.append( cv2Norm(cv2Resize(frame,resize_to)\
                                    ,self.mean, self.std).unsqueeze(0) )
        video_data = torch.cat(transformed_video)
        print('[ DEBUG ] DataGen::get_features shape of '+\
                'transformed_video {} and score {}'.format(video_data.shape, score) )
        
        output1 = torch.Tensor().cuda()
        output2 = torch.Tensor().cuda()
        frame_start = 0
        frame_end = frame_start + frame_batch_size
        with torch.no_grad():
            while frame_end < video_length:
                batch = video_data[frame_start:frame_end].cuda()
                features_mean, features_std = self.featureEx(batch)
                output1 = torch.cat((output1, features_mean), 0)
                output2 = torch.cat((output2, features_std), 0)
                frame_end += frame_batch_size
                frame_start += frame_batch_size
            last_batch = video_data[frame_start:video_length].cuda()
            features_mean, features_std = self.featureEx(last_batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            output = torch.cat((output1, output2), 1).squeeze()
        print('[ DEBUG ] DataGen::get_features shape of '+\
               'output {} and score {}'.format(output.shape, score) )
        return output, score

             
    def init_score(self, score_path):
        Info = h5py.File(score_path, 'r')
        video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                        range(len(Info['video_names'][0, :]))]
        video_names = [name.split('_')[0] for name in video_names]
        scores = Info['scores'][0, :]
        width = int(Info['width'][0])
        height = int(Info['height'][0])
        print(video_names)
        print(scores)
        print(width)
        print(height)
        score_mapper = {}
        for i in range(len(scores)):
            score_mapper[video_names[i]] = scores[i]
        return score_mapper

    def __get_vid(self, path):
        return path.split('/')[-1].split('.')[0]

    
class FeatureEx(torch.nn.Module):
    def __init__(self, pretrained_weight):
        super(FeatureEx, self).__init__()
        self.features = nn.Sequential(*list(resnet50(pretrained_weight, \
                            pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def global_std_pool2d(self, x):
        return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                dim=2, keepdim=True)

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = self.global_std_pool2d(x)
                return features_mean, features_std


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    pretrained_weight = 'models/Koniqa.pth'    
    data_set = ["KoNVid", 'LIVE', "Youtube", 'YFCC', 'IA']
    process_data = data_set[2]

    '''
    '''

    if process_data == data_set[3]:
        dg = DataGen('data/deleteLater', pretrained_weight = pretrained_weight)
        all_videos = utils.get_all_videos_from_dir('videos/LSVQ/yfcc/')
        scores = utils.get_MOS_LSVQ()
        dg.process_inp_paths(all_videos,  scores, open('deleteLater.txt','a'))

    elif process_data == data_set[4]:
        dg = DataGen('data/deleteLater', pretrained_weight = pretrained_weight)
        all_videos = utils.get_all_videos_from_dir('videos/LSVQ/ia/')
        scores = utils.get_MOS_LSVQ()
        dg.process_inp_paths(all_videos,  scores, open('deleteLater.txt','a'))

    elif process_data == data_set[2]:
        dg = DataGen('data/deleteLater', pretrained_weight = pretrained_weight)
        all_videos = utils.get_all_videos_from_dir('videos/Youtube/')
        scores = utils.get_MOS_Youtube('videos/Youtube/MOS.txt')
        dg.process_inp_paths(all_videos, scores, open('deleteLater.txt','a'))

    elif process_data == data_set[0]:
        dg = DataGen('data/deleteLater', pretrained_weight = pretrained_weight)
        all_videos = utils.obtain_all_files('videos/KoNViD-1k')
        scores = dg.init_score('videos/KoNViD-1kinfo.mat')
        dg.process_inp_paths(all_videos, scores, open('deleteLater.txt','a'))

    elif process_data == data_set[1]:
        dg = DataGen('data/deleteLater', pretrained_weight = pretrained_weight)
        all_videos = utils.obtain_all_files('videos/LIVE_VQA')
        scores = utils.get_MOS_Youtube('videos/LIVE_VQA/MOS.txt')
        dg.process_inp_paths(all_videos, scores, open('deleteLater.txt','a'))
        
