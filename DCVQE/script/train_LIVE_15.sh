set -x
set -e

epoch=150
batchsize=10
output='./LIVE_myCNNModel_15.pth'
input='./tmp/LIVE_train_15.txt'
evalinput='./tmp/LIVE_test_15.txt'
#input='data/train_CNN_features_ranking.txt'
#evalinput='data/test_CNN_features_ranking.txt'
#'models/pretrained_VQA_mean.pth'
gpu="4"
batchsize=128
reduced_size=128
experiment_writer='experiment/E150_divide_and_conquer_LIVE_exp_15.txt'
max_len=304
scale="100"

python -u train.py \
--training_data $input \
--eval_data $evalinput \
--o $output \
--epoch $epoch \
--b $batchsize \
--gpu $gpu \
--scale $scale \
--max_len $max_len \
--reduced_size $reduced_size \
--experiment_writer $experiment_writer \
--activate_leng 15
