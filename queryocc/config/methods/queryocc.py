"""QueryOcc presets: camera-only pseudo-label occupancy pre-training (queryocc-nusc),
the lidar-supervised variant (queryocc-lidar-nusc), joint pseudo+lidar supervision
(queryocc-joint-nusc), and the nuScenes+AV2 multi-dataset variant (queryocc-nusc-av2).
The release matrix over these lives in scripts/arrays/lists/release.txt.
"""

import copy

from queryocc.config.logging_configs import LoggingConfig
from queryocc.config.methods._registry import method_configs, register
from queryocc.config.trainer import TrainerConfig, TrainingConfig
from queryocc.data.datasets.zod_base import ZODBaseDatasetConfig
from queryocc.data.joint_datamodule import JointDataModuleConfig
from queryocc.data.occupancy.occupancy_sampler_gpu import LidarGPUOccupancySamplerConfig
from queryocc.data.vfm.vfm_reader import VFMReaderConfig
from queryocc.model.components.decoders.one_head import OneHeadDecoderConfig
from queryocc.model.components.image_encoders.cerveza.encoder import CervezaImageEncoderConfig
from queryocc.model.components.image_encoders.lifters.lss import LSSProConfig
from queryocc.model.components.net import NetConfig
from queryocc.model.pre_train import PreTrainConfig
from queryocc.viz.visualizer import VisualizerConfig

queryocc_nusc = register(
    "queryocc-nusc",
    TrainingConfig(
        scale_lr_with_batch_size=True,
        trainer=TrainerConfig(
            # Run exactly 12 complete epochs. A sample-budget max_steps value can stop
            # slightly before an epoch boundary after DistributedSampler padding and
            # gradient accumulation, so epoch-limited training is intentional here.
            max_steps=-1,
            max_epochs=12,
            # Validation runs only at the explicitly scheduled epochs below. Each event
            # uses 300 batches/rank, while best/final checkpoints receive a full
            # validation after training.
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            sync_batchnorm=True,
            limit_val_batches=300,
            num_sanity_val_steps=0,
            precision="bf16-mixed",
        ),
        validation_epochs=(6, 7, 8, 9, 10, 11),
        monitor_key="val/occ3d/metrics/occ3d_binary_rayiou/max_binary_rayiou",
        monitor_mode="max",
        test_after_fit_ckpt_path="best",
        test_best_and_last_after_fit=True,
        model=PreTrainConfig(
            # Tuned up from the paper's 5e-5: 7.5e-5 sits at the top of a clean
            # inverted-U in the LR sweep (5e-5 -> 23.32, 7.5e-5 -> 24.13 mean of 3 seeds,
            # 1e-4 -> 23.33 semantic RayIoU) and beat the paper's 23.6 on every seed. This is
            # the config the released queryocc-nusc checkpoint was trained with. Pass
            # --model.max_lr=5e-5 for the exact paper configuration.
            # Camera-only arm only: the derived presets below pin 5e-5, since the sweep was
            # run on this arm and the lidar/joint/AV2 arms were never tested at 7.5e-5.
            max_lr=7.5e-5,
            net=NetConfig(
                compile=True,
                lidar_encoder=False,
                image_encoder=True,
                use_one_head=True,
                one_head=OneHeadDecoderConfig(
                    occupancy=False,
                    pseudo_occupancy=True,
                    semantic_segmentation=False,
                    pseudo_semantic_segmentation=True,
                ),
                hidden_dim=160,
                use_point_contraction=True,
                semantic_segmentation_classwise_weights="class_frequency",
                bev_attention="deformable_py",
                attention_layers=4,
                swap_batch_norm_to_instance_norm=False,
                gradient_checkpointing=False,
                cerveza_encoder=CervezaImageEncoderConfig(
                    image_encoder_backbone="convnext_base",
                    image_encoder_initial_features="zeros",
                    image_encoder_points_positional_encoding="sinusoidal",
                    image_encoder_detach_previous_inputs=False,
                    lss=LSSProConfig(
                        use_point_contraction=True,
                        add_sky_depth=True,
                        depth_distribution="log_linear_blend",
                        depth_num_bins=42,
                        depth_gamma=0.3,
                        max_depth=100.0,
                    ),
                ),
            ),
            roi="surround_base",
            compute_occ3d_metrics=True,
            occupancy_sampler=LidarGPUOccupancySamplerConfig(
                # Halved from the paper's 400k (= 800k queries/sample counting positives+negatives)
                # for training speed; decoder and sampler cost scale linearly with this. Pass
                # --model.occupancy_sampler.num_query_points 400000 for exact paper reproduction.
                num_query_points=200_000,
            ),
            weight_img_depth_loss=0.5,
            logging=LoggingConfig(
                bev_features=True,
            ),
        ),
        datamodule=JointDataModuleConfig(
            datasets_to_include=("nuscenes",),
            batch_size=3,
            val_batch_size=1,
            # limit_val_batches=300/rank on four GPUs samples roughly 20% of the
            # 6,019-frame validation split. The sampler callback changes the
            # deterministic random subset at each scheduled validation epoch.
            shuffle_val=True,
            splits="original",
            dataset=ZODBaseDatasetConfig(
                make_batchable=True,
                image_size=(256, 704),
                add_missing_points=True,
                depth_maps_source="metric3d",
                av2_camera_downsample_factor=4,
                pad_images_to_same_size=True,
                n_prev_images=0,
                prev_images_max_time=0.0,
                num_future_frames_to_sample=3,
                max_future_frame_s=3.0,
                num_past_frames_to_sample=3,
                max_past_frame_s=3.0,
                n_max_offset_points_to_load_per_scan=50000,
                n_max_missing_rays_to_load_per_scan=5000,
                rotation_augmentation=True,
                rotation_range=(-180, 180),
                translation_augmentation=False,
                vfm_cfg=VFMReaderConfig(
                    sample_camera_per_timestep=True,
                ),
            ),
        ),
        allow_visualization=False,
        visualizer=VisualizerConfig(
            val_batch_interval=100,
            f_viz_bev_features=False,
            f_viz_occ_3d_bev=False,
        ),
    ),
    "QueryOcc camera-only occupancy pre-training on nuScenes using pseudo-labels. "
    "Samples a different camera per timestep.",
)

queryocc_lidar_nusc = register(
    "queryocc-lidar-nusc",
    copy.deepcopy(method_configs["queryocc-nusc"]),
    "QueryOcc occupancy pre-training on nuScenes with lidar supervision instead of pseudo point clouds.",
)
queryocc_lidar_nusc.model.max_lr = 5e-5  # untested at the camera arm's tuned 7.5e-5
queryocc_lidar_nusc.model.net.one_head.occupancy = True
queryocc_lidar_nusc.model.net.one_head.pseudo_occupancy = False
queryocc_lidar_nusc.model.net.one_head.semantic_segmentation = True
queryocc_lidar_nusc.model.net.one_head.pseudo_semantic_segmentation = False

# ---------------------------------------------------------------------------
# Joint supervision: BOTH pseudo point clouds (Metric3D depth + VFM masks) AND real aggregated lidar
# supervise the same decoder. All four heads are independent loss branches in OneHeadDecoder, so this
# is purely additive over queryocc-nusc. NB the decoder now sees ~2x the queries per sample (200k real
# + the pseudo budget); if this OOMs at the parent's bs3, drop to --datamodule.batch_size=2
# (scale_lr_with_batch_size keeps the effective LR matched).
queryocc_joint_nusc = register(
    "queryocc-joint-nusc",
    copy.deepcopy(method_configs["queryocc-nusc"]),
    "QueryOcc on nuScenes supervised by BOTH pseudo point clouds (Metric3D) AND real lidar "
    "(occupancy + semantics from each source).",
)
queryocc_joint_nusc.model.max_lr = 5e-5  # untested at the camera arm's tuned 7.5e-5
queryocc_joint_nusc.model.net.one_head.occupancy = True
queryocc_joint_nusc.model.net.one_head.semantic_segmentation = True
# pseudo_occupancy / pseudo_semantic_segmentation stay True from the parent.

# ---------------------------------------------------------------------------
# Multi-dataset scaling: nuScenes + AV2 ("sensor") jointly, real lidar as the point-cloud source (AV2
# has no Metric3D pseudo depth, so the pseudo pipeline is off — this derives from queryocc-lidar-nusc).
# Validate/test on nuScenes only (the occ3d benchmark), double the sample budget since the training
# pool roughly doubles, and a shorter warmup fraction to keep absolute warmup steps comparable. AV2
# semantic labels are mapped into the nuScenes class language by the label mapper.
# Optional via CLI: --datamodule.balanced_dataset_sampling --trainer.no-use_distributed_sampler to
# sample both datasets equally often.
queryocc_nusc_av2 = register(
    "queryocc-nusc-av2",
    copy.deepcopy(method_configs["queryocc-lidar-nusc"]),
    "QueryOcc lidar-supervised pre-training on nuScenes + AV2 jointly (val/test on nuScenes occ3d). "
    "Double budget (600k samples) for the doubled training pool.",
)
queryocc_nusc_av2.datamodule.datasets_to_include = ("nuscenes", "sensor")
queryocc_nusc_av2.datamodule.val_on_nuscenes_only = True
queryocc_nusc_av2.trainer.max_steps = 600_000
queryocc_nusc_av2.model.pct_start = 0.01
