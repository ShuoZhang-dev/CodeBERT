# pip install transformers==4.15.0 --user
# pip install tokenizers --user
# pip install nltk --user

nvidia-smi

mnt_dir="/datadisk/shuo/CodeReview"

PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NCCL_DEBUG=INFO

echo -e "import nltk\nnltk.download('punkt')" > tmp.py
python tmp.py
rm tmp.py

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} run_test_intcls.py  \
  --model_type codet5 \
  --add_lang_ids \
  --config_name ${mnt_dir}/CodeReviewer/finetuned_model/intent_cls_14_classes/checkpoints-7000-0.470 \
  --model_name_or_path ${mnt_dir}/CodeReviewer/finetuned_model/intent_cls_14_classes/checkpoints-7000-0.470 \
  --load_model_path ${mnt_dir}/CodeReviewer/finetuned_model/intent_cls_14_classes/checkpoints-7000-0.470 \
  --output_dir test_result_14_classes \
  --eval_file ${mnt_dir}/code_review_intent/data_with_intent_14_classes/msg-test.jsonl \
  --intent_dict_file intent_dict.json \
  --num_cls 14 \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 12 \
  --mask_rate 0.15 \
  --save_steps 1000 \
  --log_steps 100 \
  --train_steps 20000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --seed 2233 \
  --raw_input \
  --log_file ${mnt_dir}/CodeReviewer/test_result/intent_cls_14_classes_test/log.txt