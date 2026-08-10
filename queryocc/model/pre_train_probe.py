from typing import Dict

import torch
from torch import Tensor


class PreTrainProbe:
    """Mixin providing the dense occ3d probe for PreTrainModel.

    probe_occ3d/_create_occ3d_queries decode the fixed occ3d grid used by _step_occ3d and
    the occ_3d_bev visualization.
    """

    def probe_occ3d(self, bev_feats: Tensor) -> Dict[str, Tensor]:
        if not hasattr(self, "_cached_occ3d_queries"):
            queries = self._create_occ3d_queries()
            self._cached_occ3d_queries = queries
        else:
            queries = self._cached_occ3d_queries

        queries = queries.repeat(bev_feats.shape[0], 1, 1)

        what_to_decode = []
        if self.net.can_decode_occupancy:
            what_to_decode.append("occupancy")
        if self.net.can_decode_semantic_segmentation:
            what_to_decode.append("semantic_segmentation")

        decode_out = self.net.decode(bev_feats, queries, what_to_decode)

        return decode_out

    def _create_occ3d_queries(self) -> Tensor:
        if self.config.probe_occ3d_setting == 1:
            xs = torch.linspace(-39.8, 39.8, 200, device=self.device, requires_grad=False)
            ys = torch.linspace(-39.8, 39.8, 200, device=self.device, requires_grad=False)
            zs = torch.linspace(-0.8, 5.2, 16, device=self.device, requires_grad=False)
            xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
            queries = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1)
            queries = torch.cat((queries, torch.zeros(queries.shape[0], 1, device=self.device)), dim=-1)

        elif self.config.probe_occ3d_setting == 2:
            xs = torch.linspace(-40.0, 40.0, 200, device=self.device, requires_grad=False)
            ys = torch.linspace(-40.0, 40.0, 200, device=self.device, requires_grad=False)
            zs = torch.linspace(-1.0, 5.4, 16, device=self.device, requires_grad=False)
            xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
            queries = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1)
            queries = torch.cat((queries, torch.zeros(queries.shape[0], 1, device=self.device)), dim=-1)

        else:
            raise ValueError(f"Invalid probe occ3d setting: {self.config.probe_occ3d_setting}")

        return queries
