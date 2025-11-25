import os
import glob
import re
import math
import clip
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import plotly.express as px
import plotly.graph_objects as go


# --------------------------
# 1. Simple sinusoidal time embedding
# --------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, t):
        """
        t: shape (B,) integer timesteps
        returns: (B, dim) time embeddings
        """
        half = self.dim // 2
        # positional encoding like in transformers
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device)
            * (torch.log(torch.tensor(1e4)) / half)
        )
        # outer product: (B, half)
        sinusoid = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)
        return self.fc(emb)  # (B, dim)


# --------------------------
# 2. Simple residual block with time conditioning
# --------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)

        # map time embedding to channel dimension
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # if in/out channels differ, make a shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        """
        x: (B, C, H, W)
        t_emb: (B, time_emb_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)

        # add time embedding (reshape to channels × H × W)
        time_out = self.time_mlp(t_emb)  # (B, C_out)
        h = h + time_out[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)

        return h + self.shortcut(x)


# --------------------------
# 3. The full tiny UNet
# --------------------------
class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        # Time embeddings
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # ---- Downsample ----
        self.down1 = ResBlock(img_channels, base_channels, time_emb_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bot1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # ---- Upsample ----
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up_block1 = ResBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_block2 = ResBlock(base_channels * 2, base_channels, time_emb_dim)

        # ---- Final output ----
        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        """
        x: (B, 3, H, W)
        t: (B,) integer timesteps
        """

        # compute time embeddings
        t_emb = self.time_embedding(t)

        # ----- Down -----
        h1 = self.down1(x, t_emb)  # (B, 64, H,   W)
        h2 = self.pool1(h1)
        h2 = self.down2(h2, t_emb)  # (B, 128, H/2, W/2)

        h3 = self.pool2(h2)
        h3 = self.down3(h3, t_emb)  # (B, 256, H/4, W/4)

        # ----- Bottleneck -----
        h4 = self.bot1(h3, t_emb)

        # ----- Up -----
        u1 = self.up1(h4)  # (B, 128, H/2, W/2)
        u1 = torch.cat([u1, h2], dim=1)
        u1 = self.up_block1(u1, t_emb)

        u2 = self.up2(u1)  # (B, 64, H, W)
        u2 = torch.cat([u2, h1], dim=1)
        u2 = self.up_block2(u2, t_emb)

        return self.final_conv(u2)  # predicts noise ε(x_t,t)


# ---------------------------
# Utilities: beta schedule, sampling q(x_t | x_0, eps), saving images
# ---------------------------


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    Sample x_t from x_0 via q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps
    - x0: (B,C,H,W) in [-1,1]
    - t: LongTensor (B,) with timesteps
    - sqrt_alphas_cumprod: tensor (T,)
    - sqrt_one_minus_alphas_cumprod: tensor (T,)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    # gather the scalars for each sample in the batch
    sa = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sb = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sa * x0 + sb * noise, noise


def save_image_grid(x, path, nrow=8):
    # x in [-1,1] -> [0,1] for saving
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    grid = make_grid(x, nrow=nrow, padding=2)
    save_image(grid, path)


# ---------------------------
# Training function
# ---------------------------


def train_ddpm(
    out_dir="./ddpm_celeba",
    epochs=20,
    batch_size=128,
    lr=2e-4,
    timesteps=200,
    device=None,
    save_every=5,
    img_size=32,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # dataset + dataloader
    # transform = transforms.Compose([
    #     transforms.Resize(img_size),
    #     transforms.ToTensor(),              # [0,1]
    #     transforms.Lambda(lambda t: (t * 2.0) - 1.0)  # to [-1,1]
    # ])
    # train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ]
    )

    train_ds = torchvision.datasets.CelebA(
        root="./data", split="train", download=False, transform=transform
    )

    dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # model
    model = TinyUNet(img_channels=3, base_channels=64, time_emb_dim=256).to(device)

    # timesteps schedule
    betas = linear_beta_schedule(timesteps=timesteps).to(device)  # beta_t
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    global_step = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        for xb, _ in pbar:
            xb = xb.to(device)  # in [-1,1]
            B = xb.shape[0]

            # sample random timesteps for each image in batch uniformly from {0,...,T-1}
            t = torch.randint(0, timesteps, (B,), device=device, dtype=torch.long)

            # sample noise and x_t
            eps = torch.randn_like(xb)
            x_t, used_noise = q_sample(
                xb, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=eps
            )

            # predict eps_theta(x_t, t)
            pred = model(x_t, t)

            # loss = MSE(pred, eps)
            loss = mse(pred, used_noise)

            # backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_postfix({"loss": loss.item(), "step": global_step})

        # save checkpoint and sample images every save_every epochs
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "timesteps": timesteps,
                "betas": betas.cpu(),
            }
            torch.save(ckpt, os.path.join(out_dir, f"ckpt_epoch_{epoch+1}.pt"))
            print(f"Saved checkpoint to {out_dir}/ckpt_epoch_{epoch+1}.pt")

            # quick sampling: simple ancestral sampling (very small T for demo)
            model.eval()
            with torch.no_grad():
                # sample a small batch of noise and then do a simple reverse loop using model predictions
                n_samples = min(32, batch_size)
                x = torch.randn(n_samples, 3, img_size, img_size, device=device)
                for ti in reversed(range(timesteps)):
                    tvec = torch.full((n_samples,), ti, device=device, dtype=torch.long)
                    eps_pred = model(x, tvec)
                    alpha_bar_t = alphas_cumprod[ti].to(device)
                    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
                    # estimate x0_pred
                    x0_pred = (
                        x - sqrt_one_minus_alpha_bar_t * eps_pred
                    ) / sqrt_alpha_bar_t
                    # simple deterministic reverse (similar to DDIM with eta=0)
                    if ti > 0:
                        alpha_bar_prev = alphas_cumprod[ti - 1].to(device)
                        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)
                    else:
                        sqrt_alpha_bar_prev = torch.tensor(1.0, device=device)
                        sqrt_one_minus_alpha_bar_prev = torch.tensor(0.0, device=device)
                    x = (
                        sqrt_alpha_bar_prev * x0_pred
                        + sqrt_one_minus_alpha_bar_prev * eps_pred
                    )

                # save samples grid
                save_image_grid(
                    x.cpu(),
                    os.path.join(out_dir, f"samples_epoch_{epoch+1}.png"),
                    nrow=8,
                )
                print(f"Saved sample grid to {out_dir}/samples_epoch_{epoch+1}.png")

    print("Training finished.")
    return model, alphas_cumprod.cpu()


## CLIP RELATED DDIM
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def prepare_image_for_clip(x: torch.Tensor):
    """
    x: (B,3,H,W) in range [0, 1] or [-1,1]
    returns: (B,3,224,224) normalized tensor suitable for clip.encode_image
    """
    # to [0,1]
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)

    # resize to CLIP size using bilinear
    x_resized = F.interpolate(
        x, size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), mode="bilinear", align_corners=False
    )

    # normalize
    mean = CLIP_MEAN.to(x_resized.device)[None, :, None, None]
    std = CLIP_STD.to(x_resized.device)[None, :, None, None]
    x_norm = (x_resized - mean) / std
    return x_norm


# ---------- DDIM sampler class (extended) ----------
class DDIMSampler:
    def __init__(self, model, alphas_cumprod):
        """
        model: noise predictor (UNet) with signature model(x, t_tensor) -> eps_pred (same shape as x)
        alphas_cumprod: 1D torch tensor of length T with α̅_t (cumulative product)
        """
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.T = len(alphas_cumprod)
        # Precompute sqrt terms for convenience
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def ddim_step(self, x_t, t):
        """
        Single deterministic DDIM reverse step (eta=0).
        x_t: (B,C,H,W)
        t: int scalar (current timestep)
        Returns x_{t-1}
        """
        device = x_t.device
        # prepare t tensor for model
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=device)

        # predict noise eps_theta(x_t, t)
        eps_pred = self.model(x_t, t_tensor)  # shape matches x_t

        # compute x0_pred from x_t and eps_pred
        alpha_bar_t = self.alphas_cumprod[t].to(device)  # scalar or 0-d tensor
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        # expand tensors to (B,1,1,1) for broadcasting
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(1, 1, 1, 1)
        sqrt_1_minus_alpha_bar_t = sqrt_1_minus_alpha_bar_t.view(1, 1, 1, 1)

        x0_pred = (x_t - sqrt_1_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

        # compute alpha_bar_{t-1} (or alpha_bar_0 if t==0)
        t_prev = max(t - 1, 0)
        alpha_bar_prev = self.alphas_cumprod[t_prev].to(device)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev).view(1, 1, 1, 1)
        sqrt_1_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev).view(1, 1, 1, 1)

        # deterministic DDIM update
        x_prev = sqrt_alpha_bar_prev * x0_pred + sqrt_1_minus_alpha_bar_prev * eps_pred
        return x_prev

    def sample_from_noise(self, batch_size, shape, device):
        """
        Produce samples from pure noise using deterministic DDIM (fast).
        shape should be (C,H,W) of a single image.
        """
        # initial noise x_T
        x_t = torch.randn((batch_size,) + shape, device=device)

        # iterate t = T-1 ... 0
        for t in reversed(range(self.T)):
            x_t = self.ddim_step(x_t, t)
        return x_t  # x_0 predictions

    # ---------- DDIM inversion (seen -> seen) ----------
    def invert_image(self, x0, return_step=None, verbose=False):
        """
        Deterministically map an image x0 -> x_T (DDIM inversion).
        x0: (B,C,H,W) in [0, 1]
        Returns: x_T (same shape)
        Simplified approach: iterate forward steps t=0..T-1 using model predictions.
        """
        device = x0.device
        x = x0.clone().to(device)
        final_step = (
            (self.T - 1) if return_step is None else min(return_step, self.T - 1)
        )
        # Note: this is a simple deterministic forward pass for inversion.
        for t in range(0, final_step):  # produce x_{t+1} up to x_T-1 -> final x_T
            # predict eps at current x (model expects the corresponding timestep)
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
            eps_pred = self.model(x, t_tensor)

            # compute alpha_bar_{t+1} etc.
            alpha_bar_next = self.alphas_cumprod[t + 1].to(device)
            sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next).view(1, 1, 1, 1)
            sqrt_1_minus_alpha_bar_next = torch.sqrt(1.0 - alpha_bar_next).view(
                1, 1, 1, 1
            )

            # x_{t+1} = sqrt(alpha_bar_{t+1}) * x0 + sqrt(1 - alpha_bar_{t+1}) * eps_pred
            # (this pushes from clean image towards corresponding noisy latent)
            x = sqrt_alpha_bar_next * x0 + sqrt_1_minus_alpha_bar_next * eps_pred

            if verbose and (t % 10 == 0):
                print(f"inversion step {t+1}/{self.T-1}")
        return x

    # ---------- CLIP-guided editing ----------

    def edit_with_clip(
        self,
        x0,
        clip_model,
        clip_preprocess,
        img_desc,  # description of the input image (label → text)
        text_prompt,  # target edit description
        guidance_scale=100.0,
        guidance_lr=0.01,
        guidance_steps=1,
        edit_steps_per_t=1,
        verbose=False,
        x0islatent=False,
    ):
        """
        x0: (B,3,H,W) in [0,1]
        clip_model: pretrained CLIP model
        img_desc: string or list describing the input image (source domain)
        text_prompt: string or list describing the target edit (target domain)
        """

        device = x0.device
        batch_size = x0.shape[0]

        # --- Encode source and target texts ---
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt] * batch_size
        if isinstance(img_desc, str):
            img_desc = [img_desc] * batch_size

        text_tokens_tgt = clip.tokenize(text_prompt).to(device)
        text_tokens_src = clip.tokenize(img_desc).to(device)

        with torch.no_grad():
            text_emb_tgt = clip_model.encode_text(text_tokens_tgt)
            text_emb_src = clip_model.encode_text(text_tokens_src)
            text_emb_tgt = text_emb_tgt / text_emb_tgt.norm(dim=-1, keepdim=True)
            text_emb_src = text_emb_src / text_emb_src.norm(dim=-1, keepdim=True)
            text_dir = text_emb_tgt - text_emb_src
            text_dir = text_dir / text_dir.norm(dim=-1, keepdim=True)

        # --- Invert image to noise space ---
        if x0islatent is False:
            with torch.no_grad():
                x_t = self.invert_image(x0, verbose=verbose)
        else:
            x_t = x0.clone().to(device)

        # --- Start reverse diffusion with guidance ---
        for t in reversed(range(self.T)):
            with torch.no_grad():
                x_prev = self.ddim_step(x_t, t)

            x_guided = x_prev.clone().detach().requires_grad_(True)

            for gs in range(guidance_steps):
                img_for_clip = prepare_image_for_clip(x_guided)
                img_emb_tgt = clip_model.encode_image(img_for_clip)
                img_emb_tgt = img_emb_tgt / img_emb_tgt.norm(dim=-1, keepdim=True)

                with torch.no_grad():
                    img_emb_src = clip_model.encode_image(prepare_image_for_clip(x0))
                    img_emb_src = img_emb_src / img_emb_src.norm(dim=-1, keepdim=True)

                # --- Directional CLIP loss (Eq. 9 in paper) ---
                img_dir = img_emb_tgt - img_emb_src
                img_dir = img_dir / img_dir.norm(dim=-1, keepdim=True)

                # dir_loss = 1 - (img_dir * text_dir).sum(dim=-1).mean()
                dir_loss = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        img_dir, text_dir, dim=-1
                    ).mean()
                )

                lambda_1 = 0.3  # weight for identity loss (Eq. 11)
                # Should have a L_face as well, but omitted for simplicity
                # id_loss = lambda_1 * (x0 - x_guided).norm1(dim=-1).mean()
                id_loss = lambda_1 * torch.nn.functional.l1_loss(x_guided, x0)

                total_loss = dir_loss + id_loss

                total_loss.backward()
                # dir_loss.backward()

                grad = x_guided.grad
                with torch.no_grad():
                    x_guided = x_guided - guidance_lr * grad * guidance_scale
                    x_guided = x_guided.clamp(-1.5, 1.5).detach().requires_grad_(True)

                if verbose:
                    print(
                        f"t={t}, step={gs+1}/{guidance_steps}, dir_loss={dir_loss.item():.4f}"
                    )

            x_t = x_guided.detach()
            del x_prev, img_for_clip, img_emb_tgt, img_emb_src, dir_loss
            torch.cuda.empty_cache()

            if verbose and (t % 10 == 0):
                print(f"edited step t={t}")

        return x_t.detach()


# ---------- Step 1: precompute latents ----------
@torch.no_grad()
def precompute_latents(
    pretrained_model,
    sampler,
    dataloader,
    device,
    return_step,
    save_dir="latents_out",
    checkpoint_interval=100,  # save every N batches
    final_filename="latents_final.pt",
):
    """
    For each image in dataloader, compute deterministic DDIM forward to step `return_step`
    and save latents x_r (the noisy representation at that return_step).
    Returns list of tuples: (latent_tensor, source_image_tensor)
    - pretrained_model: used by sampler for inversion (should be the same model used during training)
    - sampler.invert_image(x0, return_step=r) should run deterministic forward to produce x_r
    """
    os.makedirs(save_dir, exist_ok=True)
    latent_list = []

    # ------------------------------------------------------------
    # Step A — Detect most recent checkpoint
    # ------------------------------------------------------------
    ckpt_pattern = re.compile(r"latents_checkpoint_batch(\d+)\.pt")
    checkpoints = []

    for fname in os.listdir(save_dir):
        match = ckpt_pattern.match(fname)
        if match:
            batch_num = int(match.group(1))
            checkpoints.append((batch_num, fname))

    # Find latest checkpoint if any
    if len(checkpoints) > 0:
        checkpoints.sort(key=lambda x: x[0])  # sort by batch_num
        last_batch, last_ckpt_name = checkpoints[-1]
        last_ckpt_path = os.path.join(save_dir, last_ckpt_name)

        print(f"[Resume] Found checkpoint: {last_ckpt_name} (completed batch {last_batch})")
        latent_list = torch.load(last_ckpt_path)
        start_batch = last_batch  # 1-indexed
    else:
        print("[Resume] No checkpoint found. Starting fresh.")
        latent_list = []
        start_batch = 0

    pretrained_model.eval()
    # ------------------------------------------------------------
    # Step B — Resume dataloader iteration
    # ------------------------------------------------------------
    # We skip ahead by simply ignoring the first `start_batch` batches.
    print(f"[Resume] Skipping first {start_batch} batches.")
    for batch_idx, batch in enumerate(dataloader):
        if start_batch >= 2500:
            break  # Limit for memory/storage
        if batch_idx < start_batch:
            continue
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
        if isinstance(batch, (list, tuple)):
            x0 = batch[0].to(device)  # (B,3,H,W) in [-1,1]
        else:
            x0 = batch.to(device)
        # invert -> latent at step r
        # Implement or call a sampler.invert_image that supports a return_step argument.
        x_r = sampler.invert_image(x0, return_step=return_step)  # (B,3,H,W)
        # store per-sample
        for i in range(x_r.shape[0]):
            latent_list.append((x_r[i].cpu(), x0[i].cpu()))

        # ---------- checkpoint save ----------
        if checkpoint_interval is not None:
            if (batch_idx + 1) % checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    save_dir, f"latents_checkpoint_batch{batch_idx+1}.pt"
                )
                torch.save(latent_list, ckpt_path)
                print(f"[Checkpoint] Saved {len(latent_list)} samples → {ckpt_path}")
                # ------------------------------------------------------------
                # <<< NEW >>> Keep only last 2 checkpoints
                # ------------------------------------------------------------
                # Re-scan checkpoints
                checkpoints = []
                for fname in os.listdir(save_dir):
                    m = ckpt_pattern.match(fname)
                    if m:
                        batch_num = int(m.group(1))
                        checkpoints.append((batch_num, fname))

                # Sort by batch number (oldest → newest)
                checkpoints.sort(key=lambda x: x[0])

                # If more than 2, delete oldest ones
                while len(checkpoints) > 2:
                    old_batch, old_file = checkpoints.pop(0)
                    old_path = os.path.join(save_dir, old_file)
                    try:
                        os.remove(old_path)
                        print(f"[Checkpoint Cleanup] Removed old checkpoint: {old_file}")
                    except FileNotFoundError:
                        pass
                # ------------------------------------------------------------

    # ---------- final save ----------
    final_path = os.path.join(save_dir, final_filename)
    torch.save(latent_list, final_path)
    print(f"[Final Save] Saved {len(latent_list)} total samples → {final_path}")
    return latent_list


# ---------- Directional CLIP loss ----------
def directional_clip_loss(clip_model, x, x_ref, text_ref_emb, text_tgt_emb):
    """
    x: (B,3,H,W) in [-1,1]
    x_ref: (B,3,H,W) reference/source images
    text_ref_emb, text_tgt_emb: (B, D) text embeddings precomputed for batch (normalized)
    Returns: scalar loss (torch.tensor)
    """
    device = x.device
    # compute image embeddings
    x_clip = prepare_image_for_clip(x)
    x_ref_clip = prepare_image_for_clip(x_ref)
    img_emb = clip_model.encode_image(x_clip)  # (B, D)
    ref_emb = clip_model.encode_image(x_ref_clip)  # (B, D)

    # normalize
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
    text_ref_emb = text_ref_emb.to(device)
    text_tgt_emb = text_tgt_emb.to(device)

    text_dir = text_tgt_emb - text_ref_emb
    text_dir = text_dir / text_dir.norm(dim=-1, keepdim=True)
    img_dir = img_emb - ref_emb
    img_dir = img_dir / img_dir.norm(dim=-1, keepdim=True)

    loss = 1.0 - torch.nn.functional.cosine_similarity(img_dir, text_dir, dim=-1).mean()
    # compute directions
    # d_img = img_emb - ref_emb
    # d_text = text_tgt_emb - text_ref_emb
    #
    # # normalize directions
    # d_img = d_img / (d_img.norm(dim=-1, keepdim=True) + 1e-7)
    # d_text = d_text / (d_text.norm(dim=-1, keepdim=True) + 1e-7)
    #
    # cos_sim = (d_img * d_text).sum(dim=-1)   # (B,)
    # loss = (1.0 - cos_sim).mean()
    return loss


# ---------- Identity loss (L2) ----------
def identity_loss(x_generated, x_ref):
    # simple pixel L2 (you can replace with face-id loss for faces)
    lambda_1 = 0.3  # weight for identity loss (Eq. 11)
    # Should have a L_face as well, but omitted for simplicity
    id_loss = torch.nn.functional.l1_loss(x_generated, x_ref)

    return id_loss
    # return F.mse_loss(x_generated, x_ref)


# ---------- Step 2: GPU-efficient fine-tuning ----------
def gpu_efficient_finetune(
    pretrained_model,
    sampler,
    latents,
    clip_model,
    optimizer,
    text_ref,
    text_tgt,
    device,
    return_step,
    gen_steps,
    finetune_iters,
    lambda_dir=1.0,
    lambda_id=0.3,
    verbose=True,
    ckpt_dir="finetune_ckpts",  # NEW
    ckpt_every=10,  # save every N outer iters
):
    """
    pretrained_model: the model copy to fine-tune (should be a clone if you want to keep original)
    sampler: DDIMSampler bound to pretrained_model (it uses model(x,t) internally)
    latents: list of (x_r_cpu, x0_cpu) tuples returned by precompute_latents
    clip_model: CLIP model already moved to device
    optimizer: optimizer over pretrained_model parameters
    text_ref, text_tgt: strings or list of strings (one per sample). We'll tokenize & encode to embeddings.
    return_step: step index r (start time for reverse DDIM)
    gen_steps: number of reverse steps to run (like # of timesteps from r -> 0)
    finetune_iters: number of iterations over dataset (outer loop)
    lambda_dir, lambda_id: weights for directional CLIP loss and identity loss
    """
    import os

    os.makedirs(ckpt_dir, exist_ok=True)

    # Precompute text embeddings (repeat if necessary)
    # If text_ref/text_tgt are single strings, broadcast per batch later.
    clip_model.eval()
    # If the text inputs are single strings, we keep them and expand per batch later.
    text_ref_token = (
        clip.tokenize([text_ref]).to(device)
        if isinstance(text_ref, str)
        else clip.tokenize(text_ref).to(device)
    )
    text_tgt_token = (
        clip.tokenize([text_tgt]).to(device)
        if isinstance(text_tgt, str)
        else clip.tokenize(text_tgt).to(device)
    )
    with torch.no_grad():
        text_ref_emb_base = clip_model.encode_text(text_ref_token)  # (1,D) or (B,D)
        text_tgt_emb_base = clip_model.encode_text(text_tgt_token)
        text_ref_emb_base = text_ref_emb_base / text_ref_emb_base.norm(
            dim=-1, keepdim=True
        )
        text_tgt_emb_base = text_tgt_emb_base / text_tgt_emb_base.norm(
            dim=-1, keepdim=True
        )

    pretrained_model.train()
    gen = torch.Generator(device=device).manual_seed(42)

    # main outer loop
    for it in range(finetune_iters):
        if verbose:
            print(f"Fine-tune iter {it+1}/{finetune_iters}")
        # iterate over latents (can randomize order)
        for idx, (x_r_cpu, x0_cpu) in enumerate(latents):
            # load single sample to device
            x_t = x_r_cpu.to(device).unsqueeze(0)  # shape (1,3,H,W)
            x_ref = x0_cpu.to(device).unsqueeze(0)

            # If text tokens are singletons, reuse base embeddings
            text_ref_emb = (
                text_ref_emb_base
                if text_ref_emb_base.shape[0] > 1
                else text_ref_emb_base.repeat(1, 1).squeeze(0).unsqueeze(0)
            )

            # If multiple target embeddings exist, pick one at random (DiffusionCLIP style)
            if text_tgt_emb_base.shape[0] > 1:
                rand_idx = torch.randint(
                    0, text_tgt_emb_base.shape[0], (1,), device=device, generator=gen
                )
                text_tgt_emb = text_tgt_emb_base[rand_idx].unsqueeze(0)
            else:
                text_tgt_emb = text_tgt_emb_base.unsqueeze(0)
            # text_tgt_emb = (
            #     text_tgt_emb_base
            #     if text_tgt_emb_base.shape[0] > 1
            #     else text_tgt_emb_base.repeat(1, 1).squeeze(0).unsqueeze(0)
            # )

            # For GPU-efficient: iterate timesteps and do loss+step each t
            # We assume sampler uses the same alphas and that t indexes are valid
            # run reverse DDIM from t = return_step down to 0 (or gen_steps steps)
            # We'll map steps: t_idx values should be chosen consistent with how you precomputed x_r
            for step_i in range(gen_steps):  # step_i = 0..gen_steps-1
                # compute current timestep index (e.g., t = return_step - step_i * stride)
                # For simplicity assume sequential integer timesteps
                t = return_step - step_i
                if t < 0:
                    break

                # Using the model's prediction (note: sampler.ddim_step should call model internally)
                # We compute x_prev (x_{t-1}) using sampler but ensure graph kept for gradients
                # sampler.ddim_step calls model(x_t, t) and returns x_prev
                x_prev = sampler.ddim_step(
                    x_t, t
                )  # this creates graph back to model parameters

                # Predict current clean image x0_pred using standard reconstruction formula
                # (we can reuse code inside sampler or recompute eps_pred, alpha bars here)
                # For readability recompute eps_pred & x0_pred:
                t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
                eps_pred = pretrained_model(x_t, t_tensor)  # (1,3,H,W)
                alpha_bar_t = sampler.alphas_cumprod[t].to(device).view(1, 1, 1, 1)
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
                x0_pred = (
                    x_t - sqrt_one_minus_alpha_bar_t * eps_pred
                ) / sqrt_alpha_bar_t

                # compute losses
                dir_loss = directional_clip_loss(
                    clip_model, x0_pred, x_ref, text_ref_emb, text_tgt_emb
                )
                id_loss = identity_loss(x0_pred, x_ref)

                loss = lambda_dir * dir_loss + lambda_id * id_loss

                # gradient step: backward & optimizer.step immediately (GPU-efficient)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # detach x_prev and use as next x_t (no graph kept)
                x_t = x_prev.detach()

            # optionally print progress
            if verbose and (idx % 10 == 0):
                print(
                    f" sample {idx}/{len(latents)} loss: {loss.item():.4f} dir: {dir_loss.item():.4f} id: {id_loss.item():.4f}"
                )

            # ───────────────────────────────────────────────────────────────
            # CHECKPOINT SAVING
            # ───────────────────────────────────────────────────────────────
            if ckpt_every is not None and (idx + 1) % ckpt_every == 0:
                ckpt_path = f"{ckpt_dir}/ft_iter_{it+1}_sample_{idx+1}.pt"
                torch.save(
                    {
                        "iter": it + 1,
                        "sample_idx": idx + 1,
                        "model": pretrained_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                if verbose:
                    print(f"Saved checkpoint ➜ {ckpt_path}")
                # ────────────────────────────────────────────────
                # PRUNE OLD CHECKPOINTS (keep only newest N)
                # ────────────────────────────────────────────────
                keep_last = 2  # keep only last N checkpoints
                ckpts = sorted(
                    glob.glob(f"{ckpt_dir}/ft_iter_*_sample_*.pt"),
                    key=os.path.getmtime,
                    reverse=True,  # newest first
                )

                if len(ckpts) > keep_last:
                    old_ckpts = ckpts[keep_last:]  # everything except newest N
                    for old in old_ckpts:
                        try:
                            os.remove(old)
                            if verbose:
                                print(f"Removed old checkpoint → {old}")
                        except OSError:
                            pass

    # save final
    final_path = f"{ckpt_dir}/final_finetuned.pt"
    torch.save(
        {
            "iter": finetune_iters,
            "model": pretrained_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        final_path,
    )

    if verbose:
        print(f"\nFinal fine-tuned model saved to {final_path}")
    return pretrained_model


def finetune_on_celeba(
    pretrained_model_path="./models/celeba/ckpt_epoch_10.pt",
    batch_size=30,
    return_step=350,
    gen_steps=50,
    finetune_iters=5,
    lambda_dir=1.0,
    lambda_id=0.3,
    lr=2e-3,
    ckpt_dir="finetune_ckpts",
    ckpt_every=1,
):
    """
    Full training pipeline:
    1. Precompute latents x_r for every CelebA image.
    2. Run your gpu_efficient_finetune() on those latents.
    """
    image_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load a pretrained model
    ckpt = torch.load(pretrained_model_path, map_location=device)
    pretrained_model = TinyUNet(img_channels=3, base_channels=64, time_emb_dim=256).to(
        device
    )
    pretrained_model.load_state_dict(ckpt["model_state_dict"])
    timesteps = ckpt["timesteps"]
    betas = ckpt["betas"].to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sampler = DDIMSampler(pretrained_model.to(device), alphas_cumprod.to(device))

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    train_ds = torchvision.datasets.CelebA(
        root="./data",
        split="train",
        target_type="attr",
        download=False,
        transform=transform,
    )
    dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 1) Compute latents (DDIM forward)
    # ------------------------------------------------------------------
    print("Precomputing latents...")
    if os.path.exists("latents_out/latents_final.pt"):
        print("Found existing latents file. Loading...")
        latents = torch.load("latents_out/latents_final.pt")
    else:
        latents = precompute_latents(
            pretrained_model=pretrained_model,
            sampler=sampler,
            dataloader=dataloader,
            device=device,
            return_step=return_step,
        )
    print(f"Latents computed: {len(latents)} samples")

    # ------------------------------------------------------------------
    # 2) Run YOUR gpu-efficient fine-tuning
    # ------------------------------------------------------------------
    text_ref = "face"
    text_tgt = [
        "angry",
        "beard",
        "smile",
        "anime",
        "gogh style",
    ]
    gpu_efficient_finetune(
        pretrained_model=pretrained_model,
        sampler=sampler,
        latents=latents,
        clip_model=clip_model,
        optimizer=optimizer,
        text_ref=text_ref,
        text_tgt=text_tgt,
        device=device,
        return_step=return_step,
        gen_steps=gen_steps,
        finetune_iters=finetune_iters,
        lambda_dir=lambda_dir,
        lambda_id=lambda_id,
        verbose=True,
        ckpt_dir=ckpt_dir,
        ckpt_every=ckpt_every,
    )

    return pretrained_model


def edit_image(
    x0,
    S_for=200,
    S_gen=199,
    unet_path="./models/celeba/ckpt_epoch_10.pt",
    finetuned_path="./finetune_ckpts/final_finetuned.pt",
    prompt_style="beard",
    image_size=128,
    img_desc="face",
    workaround=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = x0.clone().to(device)
    # Load a pretrained model
    ckpt = torch.load(unet_path, map_location=device)
    pretrained_model = TinyUNet(img_channels=3, base_channels=64, time_emb_dim=256).to(
        device
    )
    pretrained_model.load_state_dict(ckpt["model_state_dict"])
    timesteps = ckpt["timesteps"]
    betas = ckpt["betas"].to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sampler = DDIMSampler(pretrained_model.to(device), alphas_cumprod.to(device))

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Load fine-tuned model
    if workaround:
        finetuned_model = pretrained_model
    else:
        finetuned_ckpt = torch.load(finetuned_path, map_location=device)
        finetuned_model = TinyUNet(img_channels=3, base_channels=64, time_emb_dim=256).to(
            device
        )
        finetuned_model.load_state_dict(finetuned_ckpt["model"])

    # =====================================================================
    # 1) INVERSION — produce x_r using pretrained model
    # =====================================================================
    with torch.no_grad():
        x_r = sampler.invert_image(x0, return_step=S_for)

    # -----------------------------------------------------
    # 2) GENERATION: x_r → edited image using finetuned model
    # -----------------------------------------------------
    # Original image is in [0, 1] -> return a edited image in [0, 1]
    xt = x_r.clone()
    aux_sampler = DDIMSampler(finetuned_model.to(device), alphas_cumprod.to(device))
    # edited = aux_sampler.edit_with_clip(
    #     xt,
    #     clip_model,
    #     clip_preprocess,
    #     img_desc,
    #     prompt_style,
    #     guidance_scale=100.0,
    #     guidance_lr=2e-3,
    #     guidance_steps=1,
    #     verbose=True,
    #     x0islatent=True,
    # )

    # timesteps from S_for → 0
    tau = torch.linspace(S_for, 0, steps=S_gen).long().to(device)

    for i in range(S_gen - 1):
        t = tau[i].item()
        t_prev = tau[i + 1].item()

        t_tensor = torch.tensor([t], device=device)

        # ε̂_finetuned(x_t, t, ref, tgt)
        with torch.no_grad():
            eps = finetuned_model(xt, t_tensor)

        # DDIM update (deterministic)
        ab_t = alphas_cumprod[t]
        ab_prev = alphas_cumprod[t_prev]

        sqrt_ab_t = torch.sqrt(ab_t).view(1,1,1,1)
        sqrt_1m_t = torch.sqrt(1 - ab_t).view(1,1,1,1)

        sqrt_ab_prev = torch.sqrt(ab_prev).view(1,1,1,1)
        sqrt_1m_prev = torch.sqrt(1 - ab_prev).view(1,1,1,1)

        x0_pred = (xt - sqrt_1m_t * eps) / sqrt_ab_t

        xt = sqrt_ab_prev * x0_pred + sqrt_1m_prev * eps

    # convert back to image space
    edited = (xt.clamp(-1, 1) + 1) / 2

    return edited

# def edit_image_with_clip(
#     img, model_path="./models/celeba/ckpt_epoch_10.pt", prompt_style="sketch"
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Load a pretrained model
#     ckpt_path = model_path
#     ckpt = torch.load(ckpt_path, map_location=device)
#     model = TinyUNet(img_channels=3, base_channels=64, time_emb_dim=256).to(device)
#     model.load_state_dict(ckpt["model_state_dict"])
#     betas = ckpt["betas"].to(device)
#     alphas = 1.0 - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)
#     sampler = DDIMSampler(model.to(device), alphas_cumprod.to(device))
#
#     clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
#
#     transform = transforms.Compose(
#         [
#             transforms.Resize(image_size),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#         ]
#     )
#
#     x0 = train_ds[0][0].unsqueeze(0).to(device)
#
#     edited = sampler.edit_with_clip(
#         x0.to(device),
#         clip_model,
#         clip_preprocess,
#         prompt_sytle,
#         guidance_scale=100.0,
#         guidance_lr=0.01,
#         guidance_steps=1,
#         verbose=True,
#     )
#     return edited


# %%
if __name__ == "__main__":
    image_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    pretrained_model_path = "./models/celeba/ckpt_epoch_10.pt"
    finetuned_model_path = "./finetune_ckpts/final_finetuned.pt"
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    train_ds = torchvision.datasets.CelebA(
        root="./data",
        split="train",
        target_type="attr",
        download=False,
        transform=transform,
    )

    x0 = train_ds[0][0].unsqueeze(0).to(device)
    img, label = train_ds[0]
    px.imshow(img.permute(1, 2, 0).numpy())
    img = img.unsqueeze(0)


    edited = edit_image(
        img,
        S_for=199,
        S_gen=30000,
        unet_path=pretrained_model_path,
        finetuned_path=finetuned_model_path,
        prompt_style="beard",
        image_size=image_size,
        img_desc="face",
        workaround=True
    )
    px.imshow(edited.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255)
    edited

    # Fine tune
    # finetuned_model = finetune_on_celeba(
    #     pretrained_model_path="./models/celeba/ckpt_epoch_10.pt",
    #     batch_size=30,
    #     return_step=199,
    #     gen_steps=50,
    #     finetune_iters=5,
    #     lambda_dir=1.0,
    #     lambda_id=0.3,
    #     lr=2e-3,
    #     ckpt_dir="finetune_ckpts",
    #     ckpt_every=10,
    # )
    # print("Fine-tuning completed.")
    # Load a pretrained model
    ckpt_path = "./models/celeba/ckpt_epoch_10.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    pretrained_model = TinyUNet(img_channels=3, base_channels=64, time_emb_dim=256).to(
        device
    )
    pretrained_model.load_state_dict(ckpt["model_state_dict"])
    timesteps = ckpt["timesteps"]
    betas = ckpt["betas"].to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # model, alphas_cumprod = train_ddpm(
    #     out_dir="./models/celeba",
    #     epochs=10,
    #     batch_size=30,
    #     lr=2e-4,
    #     timesteps=200,
    #     img_size=image_size,
    #     device=device,
    # )

    sampler = DDIMSampler(pretrained_model.to(device), alphas_cumprod.to(device))

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ]
    )

    train_ds = torchvision.datasets.CelebA(
        root="./data",
        split="train",
        target_type="attr",
        download=False,
        transform=transform,
    )

    x0 = train_ds[0][0].unsqueeze(0).to(device)
    img, label = train_ds[0]

    img_desc = "face"
    prompt_sytle = "sketch"
    edited = sampler.edit_with_clip(
        x0.to(device),
        clip_model,
        clip_preprocess,
        img_desc,
        prompt_sytle,
        guidance_scale=100.0,
        guidance_lr=0.01,
        guidance_steps=10,
        verbose=True,
    )
#
#
# %%
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# Images are in [0, 1]
# x0 = train_ds[5][0].unsqueeze(0).to(device)
original_image_display = x0.squeeze(0).cpu()
edited_image_display = edited.squeeze(0).cpu()
# original_image_display = (x0.squeeze(0).cpu() + 1) / 2
# edited_image_display = (edited.squeeze(0).cpu() + 1) / 2


def _create_fig():
    fig = make_subplots(1, 2)
    fig.add_trace(
        go.Image(z=original_image_display.permute(1, 2, 0).numpy() * 255), row=1, col=1
    )
    fig.add_trace(
        go.Image(z=edited_image_display.permute(1, 2, 0).numpy() * 255), row=1, col=2
    )
    return fig


fig = _create_fig()
fig.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image_display.permute(1, 2, 0))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(edited_image_display.permute(1, 2, 0))
plt.title(f"Edited Image ({prompt_sytle})")
plt.axis("off")
plt.show()
plt.savefig("clip_guided_editing_result.png")

# %%
import torchvision
from torchvision.datasets import CelebA
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Root directory for the dataset
data_root = "data/celeba"
# Spatial size of training images, images are resized to this size.
image_size = 128
# batch size
batch_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5])
    ]
)

train_ds = torchvision.datasets.CelebA(
    root="./data", split="train", download=False, transform=transform
)


# Convert tensors from [-1, 1] to [0, 1] for display
x0 = train_ds[2][0].unsqueeze(0).to(device)
original_image_display = (x0.squeeze(0).cpu() + 1) / 2
original_image_display = x0.squeeze(0).cpu()
# edited_image_display = (edited.squeeze(0).cpu() + 1) / 2

px.imshow(original_image_display.permute(1, 2, 0).numpy())


# %%

# Before loop:
src_tokens = clip.tokenize(["photo"] * batch_size).to(device)
with torch.no_grad():
    src_emb = clip_model.encode_text(src_tokens)
    src_emb = src_emb / src_emb.norm(dim=-1, keepdim=True)

# Inside guidance step:
image_emb = clip_model.encode_image(img_for_clip)
image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

# Directional CLIP loss (DiffusionCLIP)
delta_i = image_emb - orig_image_emb.detach()
delta_t = text_emb - src_emb
delta_i = delta_i / delta_i.norm(dim=-1, keepdim=True)
delta_t = delta_t / delta_t.norm(dim=-1, keepdim=True)
clip_loss = 1 - (delta_i * delta_t).sum(dim=-1).mean()


# Optionally
clip_loss = 0.5 * (1 - (delta_i * delta_t).sum(dim=-1).mean()) + 0.5 * (
    1 - (image_emb * text_emb).sum(dim=-1).mean()
)
