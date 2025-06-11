#/bin/bash
source .venv/bin/activate

model_type="vgcap"
model_name="debug-${model_type}"
checkpoint_location="checkpoints"

# python train.py --id "${model_name}" \
# 	--caption_model "${model_type}" \
# 	--checkpoint_path "${checkpoint_location}" \
# 	--input_json "/home/henry/Datasets/coco/cocotalk.json" \
# 	--input_fc_dir "/home/henry/Datasets/coco/butd_fc/" \
# 	--input_att_dir "/home/henry/Datasets/coco/butd_att/" \
# 	--input_box_dir "/home/henry/Datasets/coco/butd_box/" \
# 	--sg_vocab_path "/home/henry/Datasets/coco/coco_pred_sg_rela.npy" \
# 	--sg_data_dir "/home/henry/Datasets/coco/coco_cmb_vrg_final/" \
# 	--sg_geometry_dir "/home/henry/Datasets/coco/geometry_iou-iou0.2-dist0.5-undirected/" \
#     --input_rel_box_dir "/home/henry/Datasets/coco/cocobu_box_relative/" \
# 	--input_label_h5 "/home/henry/Datasets/coco/cocotalk_label.h5" \
# 	--label_smoothing 0.0 \
# 	--batch_size 4 \
# 	--loader_num_workers 4 \
# 	--learning_rate 3e-4 \
# 	--num_layers 4 \
# 	--input_encoding_size 512 \
# 	--rnn_size 2048 \
# 	--learning_rate_decay_start 3 \
# 	--learning_rate_decay_rate 0.5 \
# 	--scheduled_sampling_start 0 \
# 	--save_checkpoint_every 10000 \
# 	--language_eval 1 \
# 	--val_images_use 5000 \
# 	--max_epochs 1 \
# 	--noamopt_warmup 33000 \
# 	--use_box 1 \
# 	--sg_label_embed_size 512 \
#     --semantic_embedding_size 512 \
#     --geometry_embedding_size 512 \
#     --structure_embedding_size 512 \
# 	--seq_per_img 5 \
# 	--use_warmup \
# 	--cached_tokens "/home/henry/Datasets/coco/coco-train-idxs" \

# Copy model for SCST
bash scripts/copy_model.sh "${checkpoint_location}" "${model_name}" "${model_name}_rl"

# python train.py --id "${model_name}_rl" \
# 	--caption_model "${model_type}" \
# 	--checkpoint_path "${checkpoint_location}" \
# 	--start_from "${checkpoint_location}" \
# 	--input_json "/home/henry/Datasets/coco/cocotalk.json" \
# 	--input_fc_dir "/home/henry/Datasets/coco/butd_fc/" \
# 	--input_att_dir "/home/henry/Datasets/coco/butd_att/" \
# 	--input_box_dir "/home/henry/Datasets/coco/butd_box/" \
# 	--sg_vocab_path "/home/henry/Datasets/coco/coco_pred_sg_rela.npy" \
# 	--sg_data_dir "/home/henry/Datasets/coco/coco_cmb_vrg_final/" \
# 	--sg_geometry_dir "/home/henry/Datasets/coco/geometry_iou-iou0.2-dist0.5-undirected/" \
#     --input_rel_box_dir "/home/henry/Datasets/coco/cocobu_box_relative/" \
# 	--input_label_h5 "/home/henry/Datasets/coco/cocotalk_label.h5" \
# 	--label_smoothing 0.0 \
# 	--batch_size 2 \
# 	--learning_rate 4e-5 \
# 	--num_layers 4 \
# 	--input_encoding_size 512 \
# 	--rnn_size 2048 \
# 	--learning_rate_decay_start 17  \
# 	--learning_rate_decay_rate 0.8  \
# 	--scheduled_sampling_start 0 \
# 	--save_checkpoint_every 3000 \
# 	--language_eval 1 \
# 	--val_images_use 5000 \
# 	--self_critical_after 1 \
# 	--max_epochs 58 \
# 	--loader_num_workers 4 \
# 	--sg_label_embed_size 512 \
# 	--seq_per_img 5 \
# 	--use_box 1 \
# 	--cached_tokens "/home/henry/Datasets/coco/coco-train-idxs" \


python eval.py --id "${model_name}" \
	--model "${checkpoint_location}/model-${model_name}-best.pth" \
	--infos_path "${checkpoint_location}/infos_${model_name}-best.pkl" \
	--input_json "/home/henry/Datasets/coco/cocotalk.json" \
	--input_fc_dir "/home/henry/Datasets/coco/butd_fc/" \
	--input_att_dir "/home/henry/Datasets/coco/butd_att/" \
	--input_box_dir "/home/henry/Datasets/coco/butd_box/" \
	--sg_vocab_path "/home/henry/Datasets/coco/coco_pred_sg_rela.npy" \
	--sg_data_dir "/home/henry/Datasets/coco/coco_cmb_vrg_final/" \
	--sg_geometry_dir "/home/henry/Datasets/coco/geometry_iou-iou0.2-dist0.5-undirected/" \
    --input_rel_box_dir "/home/henry/Datasets/coco/cocobu_box_relative/" \
	--input_label_h5 "/home/henry/Datasets/coco/cocotalk_label.h5" \
	--language_eval 1 \
	--sg_label_embed_size 512 \
