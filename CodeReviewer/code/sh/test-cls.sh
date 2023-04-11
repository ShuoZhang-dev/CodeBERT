# batch size 6 for 16 GB GPU

mnt_dir="/datadisk/shuo/CodeReview"

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
  --model_name_or_path microsoft/codereviewer \
  --output_dir ../../save/gen \
  --load_model_path ${mnt_dir}/CodeReviewer/official_finetuned_checkpoint/quality_estimation \
  --output_dir ${mnt_dir}/CodeReviewer/test_result/cls \
  --eval_file ${mnt_dir}/CodeReviewer/data/Diff_Quality_Estimation/cls-test.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 16 \
  --mask_rate 0.15 \
  --save_steps 4000 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233