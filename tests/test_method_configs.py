"""Guard tests for the method-config registry.

Run: pytest tests/test_method_configs.py
"""

from queryocc.config.methods import WANDB_TRACK_BY_MODULE, method_configs
from queryocc.config.methods._registry import method_modules

RELEASE_PRESETS = ["queryocc-nusc", "queryocc-lidar-nusc", "queryocc-joint-nusc", "queryocc-nusc-av2"]


class TestRegistry:
    def test_exactly_the_release_presets(self):
        assert sorted(method_configs) == sorted(RELEASE_PRESETS)

    def test_every_preset_has_a_group(self):
        for n in RELEASE_PRESETS:
            assert method_configs[n].wandb_group == "QueryOcc", n
            assert method_modules[n] == "queryocc", n

    def test_track_map_covers_all_modules(self):
        used_modules = set(method_modules.values())
        unmapped = used_modules - set(WANDB_TRACK_BY_MODULE)
        assert unmapped == set(), f"config modules without a track mapping: {unmapped}"


class TestReleasePresets:
    def test_supervision_sources(self):
        pseudo = method_configs["queryocc-nusc"].model.net.one_head
        assert pseudo.pseudo_occupancy and pseudo.pseudo_semantic_segmentation
        assert not pseudo.occupancy and not pseudo.semantic_segmentation

        lidar = method_configs["queryocc-lidar-nusc"].model.net.one_head
        assert lidar.occupancy and lidar.semantic_segmentation
        assert not lidar.pseudo_occupancy and not lidar.pseudo_semantic_segmentation

        joint = method_configs["queryocc-joint-nusc"].model.net.one_head
        assert joint.occupancy and joint.semantic_segmentation
        assert joint.pseudo_occupancy and joint.pseudo_semantic_segmentation

    def test_av2_arm(self):
        c = method_configs["queryocc-nusc-av2"]
        assert c.datamodule.datasets_to_include == ("nuscenes", "sensor")
        assert c.datamodule.val_on_nuscenes_only
        assert c.trainer.max_steps == 600_000
        assert c.model.pct_start == 0.01

    def test_paper_hyperparameters(self):
        c = method_configs["queryocc-nusc"]
        assert c.model.net.hidden_dim == 160
        assert c.model.net.attention_layers == 4
        assert c.model.net.bev_attention == "deformable_py"
        assert c.model.net.cerveza_encoder.lss.depth_num_bins == 42
        assert c.model.net.cerveza_encoder.lss.depth_gamma == 0.3
        assert c.model.net.cerveza_encoder.lss.max_depth == 100.0
        assert c.trainer.max_steps == -1
        assert c.trainer.max_epochs == 12
        assert c.validation_epochs == (6, 7, 8, 9, 10, 11)
        assert c.monitor_key == "val/occ3d/metrics/occ3d_binary_rayiou/max_binary_rayiou"
        assert c.datamodule.dataset.image_size == (256, 704)
        assert c.datamodule.dataset.num_future_frames_to_sample == 3
        assert c.datamodule.dataset.num_past_frames_to_sample == 3
