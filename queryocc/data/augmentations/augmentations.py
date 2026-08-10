import torch
import torch.nn.functional as F
from torch import nn


class GridMaskGPU(nn.Module):
    def __init__(
        self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0, augment_on_val=False
    ):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate  # maximum rotation in degrees
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.augment_on_val = augment_on_val

    def forward(self, x: torch.Tensor, output_mask: bool = False) -> torch.Tensor:
        # If not training and not augmenting on val, return the original image
        # Or if not augmenting, return the original image
        do_aug = torch.rand(1, device=x.device) < self.prob
        if (not self.training and not self.augment_on_val) or not do_aug:
            if output_mask:
                return x, torch.ones_like(x)
            return x

        # If augmenting, apply the grid mask
        n, c, h, w = x.shape
        device = x.device
        x_flat = x.flatten(start_dim=0, end_dim=1)

        hh = int(1.5 * h)
        ww = int(1.5 * w)

        # Random grid size
        d = torch.randint(2, h, (1,), device=device).item()
        le = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # Create a mask of ones
        mask = torch.ones((hh, ww), device=device, dtype=x.dtype)

        st_h = torch.randint(0, d, (1,), device=device).item()
        st_w = torch.randint(0, d, (1,), device=device).item()

        if self.use_h:
            for i in range(hh // d + 1):
                s = d * i + st_h
                t = min(s + le, hh)
                if s < hh:
                    mask[s:t, :] = 0

        if self.use_w:
            for i in range(ww // d + 1):
                s = d * i + st_w
                t = min(s + le, ww)
                if s < ww:
                    mask[:, s:t] = 0

        # Random rotation
        if self.rotate > 1:
            angle = torch.empty(1, device=device).uniform_(0, self.rotate).item()
            theta = torch.tensor(
                [
                    [torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0],
                    [torch.sin(torch.deg2rad(torch.tensor(angle))), torch.cos(torch.deg2rad(torch.tensor(angle))), 0],
                ],
                device=device,
            )
            grid = F.affine_grid(theta.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0).size(), align_corners=False)
            mask = F.grid_sample(
                mask.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )[0, 0]

        # Center crop
        mask = mask[(hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(x_flat)

        if self.offset:
            random_offset = 2 * (torch.rand((h, w), device=device, dtype=x.dtype) - 0.5)
            random_offset = random_offset.expand_as(x_flat)
            x_flat = x_flat * mask + random_offset * (1 - mask)
        else:
            x_flat = x_flat * mask

        x = x_flat.view(n, c, h, w)

        if output_mask:
            mask = mask.view(n, c, h, w)
            return x, mask

        return x
