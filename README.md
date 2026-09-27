# SST: Beyond Global Alignment — Structured Semantic Transfer for Zero-Shot Hashing

Anonymous code release. This repository implements the two transfer modules described in the paper — patch-angle attribute structure distillation and attribute-angle unseen–seen geometry alignment. 

## Environment

```bash
conda create -n sst python=3.10 -y
conda activate sst
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


## Training

```bash
python SST_main.py --dataset CUB --bit 64 --gpu 0 \
    --hash_loss center_loss --init bernoulli --TGI 1 --epoch 30 \
    --caption 1 --freeze_img 1 --freeze_txt 1 \
    --aug 1 --tta 1 --batch_size 32 \
    --attr_patch 1 --attr_patch_mode pair --attr_patch_weight 2.5 \
    --attr_center 1 --attr_center_mode seen_unseen --attr_center_weight 0.2 \
    --save_path ./save/CUB_64bit
```


## License

See `LICENSE`.
