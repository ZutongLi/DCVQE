set -x
set -e

for k in $( seq 1 100 )
do
    echo "Random test "$k", now gen random train and test split......"
    #KonVid-1k
    shuf tmp/iqa_shuf_data.txt > tmp/iqa_shuf_data_15mask.txt
    head -n960 tmp/iqa_shuf_data_15mask.txt > tmp/iqa_train_15mask.txt
    tail -n240 tmp/iqa_shuf_data_15mask.txt > tmp/iqa_test_15mask.txt
    bash script/train_Kon_15_mask_iqa.sh


    # LIVE VQA
    #shuf tmp/LIVE_shuf_data.txt > tmp/LIVE_shuf_data_tmp_15.txt
    #head -n468 tmp/LIVE_shuf_data_tmp_15.txt > tmp/LIVE_train_15.txt
    #tail -n117 tmp/LIVE_shuf_data_tmp_15.txt > tmp/LIVE_test_15.txt
    #bash script/train_LIVE_15.sh



    # LSVQ test and train on non-1080
    #shuf tmp/without_1080P.txt > tmp/LSVQ_training_tmp_wo1080_15.txt
    #head -n26636 tmp/LSVQ_training_tmp_wo1080_15.txt > tmp/LSVQ_train_wo1080_15.txt
    #tail -n6659 tmp/LSVQ_training_tmp_wo1080_15.txt > tmp/LSVQ_test_wo1080_15.txt
    #bash script/train_LSVQ_wo1080_15.sh
    # LSVQ  test on 1080 train on non-1080
    #shuf tmp/without_1080P.txt > tmp/LSVQ_training_tmp_1080_15.txt
    #head -n30000 tmp/LSVQ_training_tmp_1080_15.txt > tmp/LSVQ_train_1080_15.txt
    #shuf tmp/1080P.txt > tmp/LSVQ_test_1080_15.txt
    #bash script/train_LSVQ_1080_15.sh

    # Youtube 
    #shuf tmp/Youtube_shuf_data.txt > tmp/Youtube_shuf_data_tmp_15.txt
    #head -n967 tmp/Youtube_shuf_data_tmp_15.txt > tmp/Youtube_train_15.txt
    #tail -n242 tmp/Youtube_shuf_data_tmp_15.txt > tmp/Youtube_test_15.txt
    #bash script/train_Youtube_15.sh


    ## All combine
    #shuf tmp/all_combine_lst.txt > tmp/all_combine_lst_tmp_15.txt
    #head -n2395 tmp/all_combine_lst_tmp_15.txt > tmp/all_combine_train_15.txt
    #tail -n599 tmp/all_combine_lst_tmp_15.txt > tmp/all_combine_test_15.txt
    #bash script/train_All_15.sh


done
