export WANDB_API=60cdb9ae1659a9a43c90dbcf4a16b93dcd1aa63d
export PYTHONPATH=./
python ./training/pre_training.py --config_path ./config/pre_training.json
# accelerate launch \
#     --num_processes=8 \
#     ./training/pre_training.py \
#     --config_path ./config/pre_training.json