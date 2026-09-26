# SST: Beyond Global Alignment — Structured Semantic Transfer for Zero-Shot Hashing

Anonymous code release. This repository implements the two transfer modules described in the paper — patch-angle attribute structure distillation and attribute-angle unseen–seen geometry alignment — on top of a PZSH-style BLIP + learnable-center hashing pipeline.

## Environment

```bash
conda create -n bga python=3.10 -y
conda activate bga
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data and Models

Large files are not included; download them to the locations below.


| Item                                                                                       | Source                                                                                               | Expected location                           |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Prepared dataset splits (images, file lists, precomputed BLIP targets, attribute matrices) | [Baidu Pan](https://pan.baidu.com/s/1JBlRiE9wF6bELNLSlRL4tg?pwd=6pxg)                                | `dataset/` (AWA, CUB, SUN subfolders)       |
| BLIP ViT-B/16 checkpoint                                                                   | [Salesforce](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth) | `BLIP_main/models/BLIP_base.pth`            |
| Stable Diffusion v1 (pseudo-image generation only)                                         | [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5)                                 | `models/ldm/stable-diffusion-v1/model.ckpt` |


The `dataset/` archives must contain the training lists with precomputed BLIP targets used by the contrastive objective:

- `dataset/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg_blip768F.txt`
- `dataset/CUB/CUB-last50_is_txt2img/images/train_40_with_caption_catgoryname_blip768F.txt`
- `dataset/SUN_Attribute/filetxt_500_217/train_seen_blip.txt` (plus `test_full.txt`, `database.txt`)

Query/database lists without features are already provided in this repository.

Pseudo-images for unseen classes can be regenerated with:

```bash
python get_persudo_img.py --prompt "A photo of a {name}" --outdir dataset/CUB/CUB-last50_is_txt2img/images/pseudo --n_samples 40
```



## Training

```bash
python BGA_main.py --dataset CUB --bit 64 --gpu 0 \
    --hash_loss center_loss --init bernoulli --TGI 1 --epoch 30 \
    --caption 1 --freeze_img 1 --freeze_txt 1 \
    --aug 1 --tta 1 --batch_size 32 \
    --attr_patch 1 --attr_patch_mode pair --attr_patch_weight 2.5 \
    --attr_center 1 --attr_center_mode seen_unseen --attr_center_weight 0.2 \
    --save_path ./save/CUB_64bit
```

Key options: `--dataset {AWA,CUB,SUN}`, `--bit {24,48,64,128}`, `--attr_patch_pool {mean,attention,clspatch,cls,queries}` (fine-grained per-attribute patch attention = `queries`), `--attr_center_target {presence,continuous}`, `--seed`, `--dump_attr`. Per-dataset configurations used in the paper are listed in the appendix of the submission.

All experiments use seed 42. Checkpoints, center matrices, binary codes, and the patch-branch embeddings are written to `--save_path`.

## License

See `LICENSE`.