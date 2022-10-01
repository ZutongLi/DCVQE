set -x
set -e

epoch=150
batchsize=10
output='./test_myCNNModel_15.pth'
input='tmp/iqa_train_15mask.txt'
evalinput='tmp/iqa_test_15mask.txt'
#input='data/train_CNN_features_ranking.txt'
#evalinput='data/test_CNN_features_ranking.txt'
finetune_path="NA"
activate_leng="15"
#'models/pretrained_VQA_mean.pth'
gpu=3
batchsize=120
sample=-1
ranking=False
reduced_size=128
experiment_writer='experiment/15_mask_20211101.txt'
margin="0.0"

python -u train.py \
--training_data $input \
--eval_data $evalinput \
--o $output \
--epoch $epoch \
--b $batchsize \
--gpu $gpu \
--reduced_size $reduced_size \
--experiment_writer $experiment_writer \
--activate_leng $activate_leng \
--margin $margin
