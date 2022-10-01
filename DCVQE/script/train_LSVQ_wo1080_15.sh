set -x
set -e

epoch=60
batchsize=10
output='./LSVQ_wo1080_myCNNModel_15.pth'
input='tmp/LSVQ_train_wo1080_15.txt'
evalinput='tmp/LSVQ_test_wo1080_15.txt'
gpu="0"
batchsize=128
reduced_size=128
experiment_writer='experiment/E150_divide_and_conquer_LSVQ_wo1080_exp_15.txt'
max_len=384
scale="92"
activate_leng="15"

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
--activate_leng $activate_leng
