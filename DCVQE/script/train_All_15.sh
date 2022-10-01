set -x
set -e

epoch=150
batchsize=10
output='./all_myCNNModel_15.pth'
input='tmp/all_combine_train_15.txt'
evalinput='tmp/all_combine_test_15.txt'
gpu="1"
batchsize=128
reduced_size=128
experiment_writer='experiment/E150_divide_and_conquer_all_exp_15.txt'
max_len=600
scale="5.00"
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
