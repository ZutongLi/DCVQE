import os
import numpy as np
import cv2
import pandas 

def find_best_performance(path):
    if not os.path.exists(path):
        print('[ ERROR ] path {} not init, return loss 10000.... '.format(path))
        return -1
    sroccs = []
    with open(path,'r') as handle:
        for line in handle:
            line = line.strip().split()
            line = [ float(l.split(':')[1])  for l in line ]
            sroccs.append(line[1] )
    return max(sroccs)

def check_path(path, mkdir = True):
    flag = os.path.exists(path)
    if mkdir and not flag:
        os.mkdir(path)
    return flag


def obtain_all_files(path):
    ret = []
    for root, dirs, files in os.walk(path):
        ret = files
        break
    return [os.path.join(path, r) for r in ret]



def get_all_frames(path):
    ret = []
    print('[ DEBUG] :: now reading {}'.format(path))
    cap = cv2.VideoCapture(path)
    while True:
        flag, frame = cap.read()
        if flag == False:
            break
        ret.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ))
    return np.array(ret, dtype = np.uint8)



def get_all_videos_from_dir(path):
    dirs = []
    for root, dirs, files in os.walk(path):
        _=1
        break
    dirs = [os.path.join(root, d) for d in dirs]
    print(dirs)
    all_files = []
    for dir in dirs:
        for root, dirs, files in os.walk(dir):
            if len(files) !=0:
                all_files.extend([os.path.join(root, f) for f in files])
    
    ret = []
    for af in all_files:
        if '.gstmp' in af:
            continue
        '''
        cap = cv2.VideoCapture(af)
        if cap is None:
            print('[ ERROR ] cannot read {}'.format(cap) )
            continue
        else:
        '''
        ret.append(af)
    return ret


def get_MOS_Youtube(path):
    mos_dict = {}
    has_head = True
    with open(path, 'r') as handle:
        for line in handle:
            line = line.strip().split()
            if has_head:
                has_head = False
                continue
            mos_dict[line[0]] = float( line[-1] )
    return mos_dict


def get_MOS_LSVQ():
    path = 'videos/LSVQ_labels_test_1080p.csv'
    ret = {}
    pd = pandas.read_csv(path)
    cnt = len(pd.mos)
    for i in range(cnt):
        ret[pd.name[i].split('/')[-1]] = float(pd.mos[i])

    path = 'videos/LSVQ_labels_train_test.csv' 
    pd = pandas.read_csv(path)
    cnt = len(pd.mos)
    for i in range(cnt):
        ret[pd.name[i].split('/')[-1]] = float(pd.mos[i])

    return ret

if __name__ == '__main__':
    pass
