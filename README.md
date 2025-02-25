# Scaling-Curves-of-CoT-Granularity-for-Language-Model-Generalization

generate data


step ratio is X% probability to contain this COT step

python3 LIS/data.py --file COT_DATA_0118_norecap/LIS_15_10 --length 15 --train_size 100 --test_size 1000 --number_range 50 --step_ratio 1 --recap 1 --num_processes 20
python3 threestep/data.py --file COT_DATA_0118_norecap/threestep_40_00 --length 40 --train_size 1000000 --test_size 1000 --number_range 47 --step_ratio 1 --recap 1 --num_processes 20


change the LIS to relevant task

training

torchrun --nproc_per_node=4 train1.py\
 --file ${DATA_DIR}\
 --folder LIS \
 --output_dir ${OUTPUT_DIR}\
 --maxlen 524 \
 --maxdata 524 \
 --vocab 59 \
 --num_range 50\
 --weight_decay 0.05\
 --learning_rate 1e-3\
 --drop 0.1\
 --batch_size 256\
 --epoch 1\
 --warmup 0.1\
 --dmodel 256\
 --head 16\
 --jobid $PJM_JOBID\
 --num_layer 6\
 --chain \
 --rpe \
 --sft 


eval
torchrun --nproc_per_node=2 test2.py\
--file ${DATA_DIR}\
--folder LIS\
--maxlen 20\
--maxlen_ood 20 \
--maxlen_IID1 20 \
--maxlen_IID2 20 \
--maxdata 20\
--vocab 58\
--num_range 50\
--drop 0.1\
--batch_size 50\
--dmodel 256\
--head 4\
--num_layer 6\
--rpe \
--jobid $PJM_JOBID \
--model_path ${MODEL_PATH} \
--name $NAME \
--sft


Thanks for the github repo https://github.com/guyuntian/CoT_benchmark.
