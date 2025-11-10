import os
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
    sa = sqrt_alphas_cumprod[t].view(-1,1,1,1)
    sb = sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
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
    out_dir = "./ddpm_cifar",
    epochs = 20,
    batch_size = 128,
    lr = 2e-4,
    timesteps = 200,
    device = None,
    save_every = 5,
    img_size = 32
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # dataset + dataloader
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),              # [0,1]
        transforms.Lambda(lambda t: (t * 2.0) - 1.0)  # to [-1,1]
    ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # model
    model = TinyUNet(img_channels=3, base_channels=64, time_emb_dim=256).to(device)

    # timesteps schedule
    betas = linear_beta_schedule(timesteps=timesteps).to(device)                # beta_t
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
            x_t, used_noise = q_sample(xb, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=eps)

            # predict eps_theta(x_t, t)
            pred = model(x_t, t)

            # loss = MSE(pred, eps)
            loss = mse(pred, used_noise)

            # backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_postfix({'loss': loss.item(), 'step': global_step})

        # save checkpoint and sample images every save_every epochs
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'epoch': epoch,
                'timesteps': timesteps,
                'betas': betas.cpu()
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
                    x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
                    # simple deterministic reverse (similar to DDIM with eta=0)
                    if ti > 0:
                        alpha_bar_prev = alphas_cumprod[ti-1].to(device)
                        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)
                    else:
                        sqrt_alpha_bar_prev = torch.tensor(1.0, device=device)
                        sqrt_one_minus_alpha_bar_prev = torch.tensor(0.0, device=device)
                    x = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * eps_pred

                # save samples grid
                save_image_grid(x.cpu(), os.path.join(out_dir, f"samples_epoch_{epoch+1}.png"), nrow=8)
                print(f"Saved sample grid to {out_dir}/samples_epoch_{epoch+1}.png")

    print("Training finished.")
    return model, alphas_cumprod.cpu()

## CLIP RELATED DDIM
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def prepare_image_for_clip(x: torch.Tensor):
    """
    x: (B,3,H,W) in range [-1,1]
    returns: (B,3,224,224) normalized tensor suitable for clip.encode_image
    """
    # to [0,1]
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
    def invert_image(self, x0, verbose=False):
        """
        Deterministically map an image x0 -> x_T (DDIM inversion).
        x0: (B,C,H,W) in [-1,1]
        Returns: x_T (same shape)
        Simplified approach: iterate forward steps t=0..T-1 using model predictions.
        """
        device = x0.device
        x = x0.clone().to(device)
        # Note: this is a simple deterministic forward pass for inversion.
        for t in range(0, self.T - 1):  # produce x_{t+1} up to x_T-1 -> final x_T
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
        text_prompt,
        guidance_scale=100.0,
        guidance_lr=0.01,
        guidance_steps=1,
        edit_steps_per_t=1,
        verbose=False,
    ):
        """
        x0: (B,3,H,W) in [-1,1]  -- input images to edit (seen->seen case)
        clip_model: pretrained CLIP model (from clip.load)
        clip_preprocess: not strictly needed if using tensor pipeline; kept for API compatibility
        text_prompt: string or list of strings (one per batch entry)
        guidance_scale: multiplier to scale CLIP gradient (larger = stronger editing)
        guidance_lr: step size for gradient descent on x_t (small, e.g. 0.01)
        guidance_steps: number of gradient updates per DDIM step (usually 1)
        edit_steps_per_t: alias for guidance_steps
        """
        device = x0.device
        batch_size = x0.shape[0]

        # Prepare text embedding
        if isinstance(text_prompt, str):
            text_prompts = [text_prompt] * batch_size
        else:
            text_prompts = list(text_prompt)
        text_tokens = clip.tokenize(text_prompts).to(device)
        with torch.no_grad():
            text_emb = clip_model.encode_text(text_tokens)  # (B, D)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # 1) invert image to x_T
        x_t = self.invert_image(x0, verbose=verbose)  # x_T

        # 2) run reverse DDIM steps t = T-1 ... 0
        for t in reversed(range(self.T)):
            # one deterministic step
            x_prev = self.ddim_step(x_t, t)

            # CLIP guidance: n small gradient steps that nudge x_prev to increase similarity(text,image)
            # We'll update x_prev in-place using gradient ascent (maximize cosine similarity).
            # Use requires_grad on the image tensor.
            x_guided = x_prev.clone().detach().requires_grad_(True)

            for gs in range(guidance_steps):
                # Prepare image for CLIP: convert [-1,1] -> normalized clip input
                img_for_clip = prepare_image_for_clip(x_guided)  # (B,3,224,224)
                # encode image
                image_emb = clip_model.encode_image(img_for_clip)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

                # cosine similarity (B,)
                cos_sim = (image_emb * text_emb).sum(dim=-1)

                # we want to maximize cos_sim => minimize negative similarity
                clip_loss = -cos_sim.mean()

                # backward to get gradient w.r.t. x_guided
                clip_loss.backward()

                # gradient step (gradient ascent on cos_sim)
                # note: x_guided.grad shape matches x_guided
                grad = x_guided.grad
                # simple update: x_guided = x_guided - (-lr * grad) because we minimize clip_loss
                with torch.no_grad():
                    x_guided = x_guided - guidance_lr * (grad * guidance_scale)
                    # clamp to reasonable range for stability
                    x_guided = x_guided.clamp(-1.5, 1.5).detach().requires_grad_(True)

            # after guidance steps, set x_t for next iteration
            x_t = x_guided.detach()

            if verbose and (t % 10 == 0):
                print(f"edited step t={t}")

        # after loop, x_t is x_0 edited
        edited = x_t.detach()
        return edited


# %%
if __name__ == "__main__":
    model, alphas_cumprod = train_ddpm(out_dir="./", epochs=10, batch_size=128, lr=2e-4, timesteps=200)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = DDIMSampler(model.to(device), alphas_cumprod.to(device))

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    img_size = 32
    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),              # [0,1]
            transforms.Lambda(lambda t: (t * 2.0) - 1.0)  # to [-1,1]
        ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    x0 = train_ds[0][0].unsqueeze(0).to(device)


    edited = sampler.edit_with_clip(x0.to(device), clip_model, clip_preprocess, "sketch",
                                   guidance_scale=100.0, guidance_lr=0.01, guidance_steps=1, verbose=True)

# %%
import matplotlib.pyplot as plt

# Convert tensors from [-1, 1] to [0, 1] for display
x0 = train_ds[40][0].unsqueeze(0).to(device)
original_image_display = (x0.squeeze(0).cpu() + 1) / 2
edited_image_display = (edited.squeeze(0).cpu() + 1) / 2

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image_display.permute(1, 2, 0))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edited_image_display.permute(1, 2, 0))
plt.title('Edited Image (Van Gogh style)')
plt.axis('off')

plt.show()
