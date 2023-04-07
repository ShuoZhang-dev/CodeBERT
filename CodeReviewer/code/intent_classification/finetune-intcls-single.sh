# pip install transformers==4.15.0 --user
# pip install tokenizers --user
# pip install nltk --user

nvidia-smi

mnt_dir="/datadisk/shuo/CodeReview"

PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NCCL_DEBUG=INFO

echo -e "import nltk\nnltk.download('punkt')" > tmp.py
python tmp.py
rm tmp.py

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} run_finetune_intcls.py  \
  --model_type codet5 \
  --add_lang_ids \
  --train_epochs 20 \
  --config_name ${mnt_dir}/CodeReviewer/pretrained_checkpoint \
  --model_name_or_path ${mnt_dir}/CodeReviewer/pretrained_checkpoint \
  --load_model_path ${mnt_dir}/CodeReviewer/pretrained_checkpoint \
  --output_dir ${mnt_dir}/CodeReviewer/finetuned_model/intent_cls_14_classes_test \
  --train_filename ${mnt_dir}/code_review_intent/data_with_intent_14_classes_test/msg-train.jsonl \
  --dev_filename ${mnt_dir}/code_review_intent/data_with_intent_14_classes_test/msg-valid.jsonl \
  --intent_dict_file intent_dict.json \
  --num_cls 14 \
  --max_source_length 512 \
  --max_target_length 128 \
  --train_batch_size 12 \
  --learning_rate 4e-5 \
  --gradient_accumulation_steps 1 \
  --mask_rate 0.15 \
  --save_steps 1000 \
  --log_steps 100 \
  --train_steps 20000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --seed 2233 \
  --raw_input \
  --log_file ${mnt_dir}/CodeReviewer/finetuned_model/logs/intent_cls_14_classes_test.txt 
  # --from_scratch \
  # --load_steps 0