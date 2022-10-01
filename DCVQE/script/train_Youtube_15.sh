set -x
set -e

epoch=150
batchsize=10
output='./Youtube_myCNNModel_15.pth'
input='./tmp/Youtube_train_15.txt'
evalinput='./tmp/Youtube_test_15.txt'
gpu="4"
batchsize=128
reduced_size=128
experiment_writer='experiment/120_divide_and_conquer_Youtube_exp_15_margin_0.1.txt'
max_len=600
scale="4.68"
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
