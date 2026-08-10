"""Single source of truth for the pseudo-label data contract.

Everything a consumer (your training dataloader) needs to know about the
generated files is defined here: camera order, class-id space, vocabularies
and detector thresholds. The generators also embed `CAMERA_ORDER` inside
every .npz they write, so files are self-describing.
"""

# ---------------------------------------------------------------------------
# Camera order
# ---------------------------------------------------------------------------
# Canonical order used by BOTH the semantic and the depth stream in the
# default (npz) layout. One order, everywhere.
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# Legacy GaussianFlowOcc orders — only used by `--layout legacy`, which
# reproduces the exact on-disk format the original trainer reads.
# NOTE the two legacy streams use DIFFERENT camera orders; this was a
# silent-corruption hazard in the original code base and is the reason the
# default layout pins a single order.
LEGACY_MASK_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_FRONT_RIGHT",
)
LEGACY_DEPTH_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

# ---------------------------------------------------------------------------
# Class-id space (nuScenes-lidarseg "v1" ids, same as GaussianFlowOcc)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "noise",           # ignore / unlabeled
    1: "barrier",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "construction_vehicle",
    6: "motorcycle",
    7: "pedestrian",
    8: "traffic_cone",
    9: "trailer",
    10: "truck",
    11: "driveable_surface",
    12: "other_flat",
    13: "sidewalk",
    14: "terrain",
    15: "manmade",
    16: "vegetation",
    17: "free",           # occupancy-GT only; never appears in 2D masks
    18: "sky",            # 2D masks only (default --sky-label)
}
CLASS_NAME_TO_LABEL = {v: k for k, v in CLASS_NAMES.items()}

# Default label written for sky pixels. GaussianFlowOcc variants used 0
# (original, sky = ignore) or 17 (w-sky, sky = "free", which silently
# collides with the free-space id). 18 keeps the information without the
# collision; map it to whatever you want at load time.
DEFAULT_SKY_LABEL = 18

# ---------------------------------------------------------------------------
# Open-vocabulary prompts
# ---------------------------------------------------------------------------
# Keys are class names from CLASS_NAME_TO_LABEL (plus "sky"). Values are the
# text prompts sent to GroundingDINO, one forward pass per prompt
# ("separate" method). Prompts MUST be lower-case: GroundingDINO returns
# lower-cased phrases and the pipeline keeps a detection only if the decoded
# phrase string exactly equals its prompt. Hyphens are replaced by spaces
# before prompting.
VOCABULARIES = {
    # The vocabulary that produced the published GaussianFlowOcc labels
    # (grounded_sam_nusc / grounded_sam1_*): 42 prompts incl. sky.
    "gaussianflowocc": {
        "car": ["car", "vehicle", "sedan", "van", "jeep"],
        "truck": ["lorry", "truck", "tractor"],
        "trailer": ["trailer"],
        "bus": ["bus"],
        "construction_vehicle": ["excavator", "crane"],
        "bicycle": ["bicycle"],
        "motorcycle": ["motorcycle", "scooter"],
        "pedestrian": ["person", "pedestrian"],
        "traffic_cone": ["traffic-cone"],
        "barrier": ["barricade", "barrier"],
        "driveable_surface": ["highway", "street"],
        "other_flat": ["traffic-island", "delimiter"],
        "sidewalk": ["sidewalk", "walkway"],
        "terrain": ["grass", "sand", "gravel", "terrain"],
        "manmade": [
            "building", "wall", "fence", "pole", "sign", "light", "bridge",
            "billboard",
        ],
        "vegetation": ["bush", "plants", "tree"],
        "sky": ["sky"],
    },
    # Later experimental vocabulary (grounded_sam1_nusc_text-v1 /
    # mult-labels): refined car/truck prompts.
    "text-v1": {
        "car": ["car", "sedan", "van", "jeep", "suv"],
        "truck": ["pickup", "lorry", "truck", "tractor-unit"],
        "trailer": ["trailer"],
        "bus": ["bus"],
        "construction_vehicle": ["excavator", "crane"],
        "bicycle": ["bicycle"],
        "motorcycle": ["motorcycle", "scooter"],
        "pedestrian": ["person", "pedestrian"],
        "traffic_cone": ["traffic-cone"],
        "barrier": ["barricade", "barrier"],
        "driveable_surface": ["highway", "street"],
        "other_flat": ["traffic-island", "delimiter"],
        "sidewalk": ["sidewalk", "walkway"],
        "terrain": ["grass", "sand", "gravel", "terrain"],
        "manmade": [
            "building", "wall", "fence", "pole", "sign", "light", "bridge",
            "billboard",
        ],
        "vegetation": ["bush", "plants", "tree"],
        "sky": ["sky"],
    },
}

# GroundingDINO thresholds used for all published GaussianFlowOcc label sets.
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.20

# GroundingDINO input resize: short side -> DINO_SIZE, long side capped at
# DINO_MAX_SIZE. For nuScenes 900x1600 the cap binds -> DINO sees 750x1333.
# SAM always runs at the full original resolution.
DINO_SIZE = 800
DINO_MAX_SIZE = 1333

# Model checkpoints
DINO_VARIANTS = {
    "swinb": ("ShilongLiu/GroundingDINO", "groundingdino_swinb_cogcoor.pth",
              "GroundingDINO_SwinB.cfg.py"),
    "swint": ("ShilongLiu/GroundingDINO", "groundingdino_swint_ogc.pth",
              "GroundingDINO_SwinT_OGC.cfg.py"),
}
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# ---------------------------------------------------------------------------
# Metric3D
# ---------------------------------------------------------------------------
METRIC3D_HUB_REPO = "yvanyin/metric3d"
METRIC3D_DEFAULT_MODEL = "metric3d_vit_large"
# Network input canvas per backbone family (from the Metric3D docs).
METRIC3D_INPUT_SIZES = {
    "vit": (616, 1064),
    "convnext": (544, 1216),
}
# Metric3D predicts depth for a canonical camera with this focal length;
# metric depth = canonical depth * (fx_resized / CANONICAL_FOCAL).
METRIC3D_CANONICAL_FOCAL = 1000.0
# ImageNet mean/std * 255, RGB order (also used as the pad value).
METRIC3D_MEAN = (123.675, 116.28, 103.53)
METRIC3D_STD = (58.395, 57.12, 57.375)
METRIC3D_DEPTH_CLAMP = (0.0, 300.0)

# ---------------------------------------------------------------------------
# Output contract (npz layout)
# ---------------------------------------------------------------------------
# <save_path>/<scene_name>/<token>.npz
#   semantics : labels  int8    [6, H, W]   (CAMERA_ORDER)
#               scores  float16 [6, H, W]   (optional, --save-scores)
#               cam_order  <U18 [6]
#   depth     : depth   float16 [6, H, W]   (CAMERA_ORDER)
#               confidence float16 [6,H,W]  (optional, --save-confidence)
#               cam_order  <U18 [6]
# <save_path>/_meta_semantics.json / _meta_depth.json : full provenance
#   (vocabulary, thresholds, model ids, scale, command line, package version)
SEMANTICS_META_FILE = "_meta_semantics.json"
DEPTH_META_FILE = "_meta_depth.json"
