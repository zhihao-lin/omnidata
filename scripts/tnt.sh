python -m omnidata_tools.torch.demo_sdfstudio \
  --task depth --mode whole \
  --source_dir /hdd/datasets/TanksAndTempleBG/M60/rgb \
  --output_dir /hdd/datasets/TanksAndTempleBG/M60/depth 

python -m omnidata_tools.torch.demo_sdfstudio \
  --task normal --mode whole \
  --source_dir /hdd/datasets/TanksAndTempleBG/M60/rgb \
  --output_dir /hdd/datasets/TanksAndTempleBG/M60/normal