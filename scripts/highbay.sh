DATA_ROOT=/hdd/datasets/highbay/0904/8_28_23_2023-09-04-09-13-16

python -m omnidata_tools.torch.demo_sdfstudio \
  --task depth \
  --source_dir $DATA_ROOT/left/rgb \
  --output_dir $DATA_ROOT/left/depth

python -m omnidata_tools.torch.demo_sdfstudio \
  --task depth \
  --source_dir $DATA_ROOT/right/rgb \
  --output_dir $DATA_ROOT/right/depth

python -m omnidata_tools.torch.demo_sdfstudio \
  --task normal \
  --source_dir $DATA_ROOT/left/rgb \
  --output_dir $DATA_ROOT/left/normal

python -m omnidata_tools.torch.demo_sdfstudio \
  --task normal \
  --source_dir $DATA_ROOT/right/rgb \
  --output_dir $DATA_ROOT/right/normal