# batch size 6 for 16 GB GPU

mnt_dir="/datadisk/shuo/CodeReview"


# You may change the following block for multiple gpu training
MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO


bash test_nltk.sh

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_msg.py  \
  --model_name_or_path ${mnt_dir}/CodeReviewer/official_finetuned_checkpoint/comment_generation \
  --output_dir ${mnt_dir}/CodeReviewer/test_result/msg_by_intent \
  --load_model_path ${mnt_dir}/CodeReviewer/official_finetuned_checkpoint/comment_generation \
  --eval_file  ${1} \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 6 \
  --mask_rate 0.15 \
  --save_steps 1800 \
  --beam_size 10 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --raw_input
