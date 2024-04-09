pretrained_model_name_or_path="./stable-diffusion-v1-4"
instance_data_dir="./data/Teyvat/Albedo"
class_data_dir="./data/Teyvat/data"
output_dir="./weights/"
train_batch_size=4

torchrun --nproc_per_node 1 train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$pretrained_model_name_or_path  \
  --instance_data_dir=$instance_data_dir \
  --class_data_dir=$class_data_dir \
  --output_dir=$output_dir \
  --instance_prompt="a gsipt anime character" \
  --resolution=512 \
  --plugin="torch_ddp" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="a anime character" \
  --train_batch_size=$train_batch_size \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --max_train_steps=800 \
  --num_class_images=200 \
  --save_steps=1000 \
  --test_run=True 
