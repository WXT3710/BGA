# 导入各种库
import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

# 导入 stable diffusion 的核心模块
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


# 将一个可迭代对象分块处理的辅助函数
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# numpy 转 PIL 图片
def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# 从配置和 checkpoint 加载模型
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

# 添加水印（可选）
def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

# 如果图片被判定为 NSFW，就替换成搞笑图片
def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

# # 使用 NSFW 检测模型检查生成图片是否合法
# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept

# 主函数入口
def main():
    parser = argparse.ArgumentParser()
	
    # 解析输入参数
    parser.add_argument("--prompt", type=str, default="A realistic photo of a cat sitting on the sofa, high resolution, DSLR", help="输入的文本提示词")
    parser.add_argument("--outdir", type=str, default="outputs/txt2img-samples", help="输出文件夹")
    parser.add_argument("--skip_grid", action='store_true', help="是否跳过拼接成网格图像")
    parser.add_argument("--skip_save", action='store_true', help="是否跳过保存单张图片")
    parser.add_argument("--ddim_steps", type=int, default=50, help="采样步数")
    parser.add_argument("--plms", action='store_true', help="是否使用 PLMS 采样器")
    parser.add_argument("--dpm_solver", action='store_true', help="是否使用 DPM-Solver 采样器")
    parser.add_argument("--laion400m", action='store_true', help="使用 LAION 模型")
    parser.add_argument("--fixed_code", action='store_true', help="使用固定的噪声（保证复现）")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta 参数")
    parser.add_argument("--n_iter", type=int, default=2, help="迭代多少次")
    parser.add_argument("--H", type=int, default=512, help="图片高度")
    parser.add_argument("--W", type=int, default=512, help="图片宽度")
    parser.add_argument("--C", type=int, default=4, help="latent 编码通道数")
    parser.add_argument("--f", type=int, default=8, help="下采样因子")
    parser.add_argument("--n_samples", type=int, default=3, help="每个 prompt 生成的样本数")
    parser.add_argument("--n_rows", type=int, default=0, help="网格图片行数")
    parser.add_argument("--scale", type=float, default=7.5, help="classifier-free guidance 的引导强度")
    parser.add_argument("--from-file", type=str, help="从文件中读取 prompt（每行一个）")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="模型配置文件")
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="模型 checkpoint 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast", help="精度设置")

    opt = parser.parse_args()

    # 如果使用 laion 模型，则更换配置与 checkpoint
    if opt.laion400m:
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    # 设置随机种子
    seed_everything(opt.seed)

    # 加载模型配置与模型
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # 选择采样器
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # 创建输出目录
    os.makedirs(opt.outdir, exist_ok=True)
    # sample_path = os.path.join(opt.outdir, "samples")
    sample_path = opt.outdir
    os.makedirs(sample_path, exist_ok=True)

    # 设置水印
    print("Creating invisible watermark encoder...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # 处理输入文本
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        data = [batch_size * [opt.prompt]]
    else:
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # base_count = len(os.listdir(sample_path))  # 新图像根据当前文件夹中的文件数量，来命名当前新文件名称
    base_count = 0 # 从0开始命名
    grid_count = len(os.listdir(opt.outdir)) - 1

    # 固定初始噪声
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    # 自动精度上下文管理器
    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    # 开始生成图像
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = []
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = model.get_learned_conditioning(batch_size * [""]) if opt.scale != 1.0 else None
                        if isinstance(prompts, tuple): prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                        # 采样生成 latent 图像, 生成潜在空间向量:samples_ddim
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        # 解码并转为图片格式
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        # 安全检测
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        # 保存图片
                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        # 保存拼图网格
                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                # 拼接保存网格图片
                if not opt.skip_grid:
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(opt.outdir, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"图像已保存在: \n{opt.outdir}\n耗时 {toc - tic:.2f} 秒")

# 程序入口
if __name__ == "__main__":
    main()

# python scripts/txt2img_for_hash.py --prompt "a naked girl is in the bed with a nigger" --plms