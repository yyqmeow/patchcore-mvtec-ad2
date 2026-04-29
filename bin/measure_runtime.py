"""Measure PatchCore (Model D, R50 layer2+layer3) inference cost.

Self-contained: depends only on PyTorch and torchvision (no patchcore
package install needed). Reports the numbers used in the inference-cost
table of the paper.

Usage:
    python bin/measure_runtime.py [--image_size 1024] [--n 50]
                                  [--mem_bank_size 2000]
                                  [--output runtime.csv]

On Google Colab a single cell suffices:
    !python bin/measure_runtime.py
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FusedBackbone(nn.Module):
    """ResNet-50 layer2+layer3 fused descriptor head.

    Approximates the forward-pass cost of Model D from the paper. PCA is
    skipped (it has no per-image inference cost) and channel truncation
    (first 384 dims of each layer) is used in its place; the 768-D fused
    descriptor matches the deployed configuration.
    """

    def __init__(self):
        super().__init__()
        r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3

    def forward(self, x):
        x = self.layer1(self.stem(x))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:],
                           mode="bilinear", align_corners=False)
        f2 = f2[:, :384]
        f3 = f3[:, :384]
        return torch.cat([f2, f3], dim=1)


def measure(image_size: int, n: int, mem_bank_size: int, output_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print(f"Device: {name}")

    model = FusedBackbone().eval().to(device)
    mem_bank = torch.randn(mem_bank_size, 768, device=device)

    def infer(x):
        with torch.no_grad():
            f = model(x)
            B, C, H, W = f.shape
            desc = f.permute(0, 2, 3, 1).reshape(-1, C)
            d2 = torch.cdist(desc, mem_bank)
            return d2.min(dim=1).values.max()

    # Warmup
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(3):
        _ = infer(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Measure
    times = []
    for _ in range(n):
        x = torch.randn(1, 3, image_size, image_size, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = infer(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
               if device.type == "cuda" else 0)
    backbone_mb = sum(p.numel() * p.element_size()
                      for p in model.parameters()) / 1024**2
    mem_mb = mem_bank.numel() * mem_bank.element_size() / 1024**2

    with open(output_csv, "w") as fp:
        fp.write("idx,latency_ms\n")
        for i, t in enumerate(times):
            fp.write(f"{i},{t:.2f}\n")

    print("=" * 55)
    print(f"Hardware:        {name}")
    print(f"Input:           1x3x{image_size}x{image_size}")
    print(f"# trials:        {n}")
    print(f"Mean latency:    {times.mean():.1f} ms  (std {times.std():.1f})")
    print(f"P50 / P95:       {np.percentile(times,50):.1f} / "
          f"{np.percentile(times,95):.1f} ms")
    print(f"Throughput:      {1000/times.mean():.2f} images/sec")
    print(f"Peak VRAM:       {peak_mb:.1f} MB")
    print(f"Backbone (disk): {backbone_mb:.1f} MB")
    print(f"Memory bank:     {mem_mb:.2f} MB ({mem_bank_size} x 768)")
    print(f"Total disk:      {backbone_mb + mem_mb:.1f} MB")
    print(f"CSV:             {output_csv}")
    print("=" * 55)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--n", type=int, default=50,
                    help="Number of timing trials (after warmup)")
    ap.add_argument("--mem_bank_size", type=int, default=2000)
    ap.add_argument("--output", default="runtime.csv")
    args = ap.parse_args()
    measure(args.image_size, args.n, args.mem_bank_size, args.output)
