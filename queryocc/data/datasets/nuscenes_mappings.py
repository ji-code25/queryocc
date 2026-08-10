import numpy as np
import torch

NUSCENES_DYNAMIC_LABELS = [2, 3, 4, 5, 6, 7, 9, 10, 17]
NUSCENES_STATIC_LABELS = [1, 8, 11, 12, 13, 14, 15, 16, 17, 18]
NUSCENES_GROUND_LABELS = [11, 12]
NUSCENES_IDX_TO_LABELS_MAP = {
    0: "noise",
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
    17: "free",
    18: "sky",
}
NUSCENES_LABELS_TO_IDX_MAP = {v: k for k, v in NUSCENES_IDX_TO_LABELS_MAP.items()}

NUSCENES_LABEL_TO_ZOD_LABEL_MAP = {
    "noise": "Unclear",
    "barrier": "DynamicBarrier",
    "bicycle": "VulnerableRider",
    "bus": "Vehicle",
    "car": "Vehicle",
    "construction_vehicle": "Vehicle",
    "motorcycle": "VulnerableRider",
    "pedestrian": "Pedestrian",
    "traffic_cone": "TrafficGuide",
    "trailer": "Vehicle",
    "truck": "Vehicle",
    "driveable_surface": "EgoRoad",
    "other_flat": "Unclear",
    "sidewalk": "Unclear",
    "terrain": "Unclear",
    "manmade": "Unclear",
    "vegetation": "Unclear",
    "free": "Free",
    "sky": "Free",
}

NUSCENES_LABEL_TO_AV2_LABEL_MAP = {
    "noise": "UNCLEAR",
    "barrier": "CONSTRUCTION_BARREL",
    "bicycle": "BICYCLIST",
    "bus": "BUS",
    "car": "REGULAR_VEHICLE",
    "construction_vehicle": "LARGE_VEHICLE",
    "motorcycle": "MOTORCYCLIST",
    "pedestrian": "PEDESTRIAN",
    "traffic_cone": "CONSTRUCTION_CONE",
    "trailer": "VEHICULAR_TRAILER",
    "truck": "TRUCK",
    "driveable_surface": "DRIVABLE_SURFACE",
    "other_flat": "UNCLEAR",
    "sidewalk": "UNCLEAR",
    "terrain": "UNCLEAR",
    "manmade": "UNCLEAR",
    "vegetation": "UNCLEAR",
    "free": "FREE",
    "sky": "FREE",
}

NUSCENES_IDX_TO_COLOR_MAP = {
    0: [0, 0, 0],  # noise
    1: [50, 120, 255],  # barrier
    2: [245, 230, 100],  # bicycle
    3: [255, 0, 0],  # bus
    4: [245, 150, 100],  # car
    5: [255, 0, 0],  # construction_vehicle
    6: [150, 60, 30],  # motorcycle
    7: [30, 30, 255],  # pedestrian
    8: [0, 0, 255],  # traffic cone
    9: [255, 0, 0],  # trailer
    10: [180, 30, 80],  # truck
    11: [255, 0, 255],  # driveable_surface
    12: [75, 0, 175],  # other_flat
    13: [75, 0, 75],  # sidewalk
    14: [80, 240, 150],  # terrain
    15: [0, 200, 255],  # manmade
    16: [0, 175, 0],  # vegetation
    17: [245, 245, 220],  # free
    18: [173, 216, 230],  # sky
}

DEFAULT_NUSCENES_CLASSES = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    # "sky",  # it should not include sky.
)

NUSCENES_LIDARSEG_IDX_TO_NUSCENES_IDX_MAP = {
    1: 0,  # noise
    5: 0,  # noise
    7: 0,  # noise
    8: 0,  # noise
    10: 0,  # noise
    11: 0,  # noise
    13: 0,  # noise
    19: 0,  # noise
    20: 0,  # noise
    0: 0,  # noise
    29: 0,  # noise
    31: 0,  # noise
    9: 1,  # barrier
    14: 2,  # bicycle
    15: 3,  # bus
    16: 3,  # bus
    17: 4,  # car
    18: 5,  # construction vehicle
    21: 6,  # motorcycle
    2: 7,  # pedestrian
    3: 7,  # pedestrian
    4: 7,  # pedestrian
    6: 7,  # pedestrian
    12: 8,  # trafic cone
    22: 9,  # trailer
    23: 10,  # truck
    24: 11,  # driveble surface
    25: 12,  # other_flat
    26: 13,  # sidewalk
    27: 14,  # terrain
    28: 15,  # manmade
    30: 16,  # vegetation
}

NUSCENES_LABEL_TO_POINTS_PER_CLASS_MAP = {
    "noise": 280703380,
    "barrier": 7618481,
    "bicycle": 119419,
    "bus": 3744383,
    "car": 31790017,
    "construction_vehicle": 1314388,
    "motorcycle": 344496,
    "pedestrian": 1888110,
    "traffic_cone": 619847,
    "trailer": 4509696,
    "truck": 13229499,
    "driveable_surface": 259466037,
    "other_flat": 6559178,
    "sidewalk": 57234476,
    "terrain": 56375263,
    "manmade": 145714137,
    "vegetation": 100099497,
    "free": 0,
    # "sky": 259466037,  # Assuming same as driveable_surface...
}

NUSCENES_PSEUDO_LABEL_TO_POINTS_PER_CLASS_MAP = {
    "noise": 8337330322,
    "barrier": 2476035726,
    "bicycle": 35063994,
    "bus": 829313058,
    "car": 8055006711,
    "construction_vehicle": 146328328,
    "motorcycle": 41840002,
    "pedestrian": 408465249,
    "traffic_cone": 128947950,
    "trailer": 95945412,
    "truck": 2174524615,
    "driveable_surface": 25561652023,
    "other_flat": 7056693674,
    "sidewalk": 7056693674,
    "terrain": 8541355984,
    "manmade": 41254816624,
    "vegetation": 20044242467,
    "free": 0,
    "sky": 22336117861,
}

class LabelMapper:
    def __init__(self, learning_classes):
        # Add "free" to your training classes
        self.learning_classes = ("noise",) + tuple(learning_classes) + ("free",)
        self._free_idx = len(self.learning_classes) - 1
        self._noise_idx = 0

        # --- Base mappings ---
        self._nusc_idx_to_label = NUSCENES_IDX_TO_LABELS_MAP
        self._nusc_label_to_idx = NUSCENES_LABELS_TO_IDX_MAP
        self._lidarseg_idx_to_nusc_idx = NUSCENES_LIDARSEG_IDX_TO_NUSCENES_IDX_MAP
        self._nusc_idx_to_color = NUSCENES_IDX_TO_COLOR_MAP
        self._nusc_label_to_points_per_class = NUSCENES_LABEL_TO_POINTS_PER_CLASS_MAP
        self._nusc_label_to_points_per_class_pseudo = NUSCENES_PSEUDO_LABEL_TO_POINTS_PER_CLASS_MAP
        self._nusc_label_to_zod = NUSCENES_LABEL_TO_ZOD_LABEL_MAP
        self._nusc_label_to_av2 = NUSCENES_LABEL_TO_AV2_LABEL_MAP

        # --- Learning mappings ---
        self._learning_idx_to_label = {i: lbl for i, lbl in enumerate(self.learning_classes)}
        self._learning_label_to_idx = {lbl: i for i, lbl in self._learning_idx_to_label.items()}

        # Learning idx -> nusc idx
        self._learning_idx_to_nusc_idx = {
            i: self._nusc_label_to_idx[lbl] for i, lbl in self._learning_idx_to_label.items()
        }

        # Nusc idx -> learning idx (default → "noise")
        self._learning_nusc_idx_to_idx = {
            self._nusc_label_to_idx[lbl]: i for i, lbl in self._learning_idx_to_label.items()
        }
        left_over = set(self._nusc_idx_to_label.values()) - set(self.learning_classes)
        left_over_idx = [self._nusc_label_to_idx[lbl] for lbl in left_over]
        for idx in left_over_idx:
            self._learning_nusc_idx_to_idx[idx] = self._learning_label_to_idx["noise"]

        # --- Precompute LUTs ---
        self._lidarseg_to_nusc_lut = self._make_lut(self._lidarseg_idx_to_nusc_idx, default=0)
        self._nusc_to_learning_lut = self._make_lut(
            self._learning_nusc_idx_to_idx, default=self._learning_label_to_idx["noise"]
        )
        chain_map = {k: self._learning_nusc_idx_to_idx[v] for k, v in self._lidarseg_idx_to_nusc_idx.items()}
        self._lidarseg_to_learning_lut = self._make_lut(chain_map, default=self._learning_label_to_idx["noise"])

        # --- nusc idx → color LUT ---
        self._nusc_to_color_lut = self._make_color_lut(self._nusc_idx_to_color, default=[0, 0, 0])

        # --- learning idx → color LUT ---
        learning_color_map = {
            l_idx: self._nusc_idx_to_color[self._learning_idx_to_nusc_idx[l_idx]]
            for l_idx in self._learning_idx_to_nusc_idx
        }
        self._learning_to_color_lut = self._make_color_lut(learning_color_map, default=[255, 255, 255])

        # --- dynamic/static in learning space ---
        self._dynamic_learning_idxs = {self._learning_nusc_idx_to_idx[nusc_id] for nusc_id in NUSCENES_DYNAMIC_LABELS}
        self._static_learning_idxs = {self._learning_nusc_idx_to_idx[nusc_id] for nusc_id in NUSCENES_STATIC_LABELS}

        # --- ground in learning space ---
        self._ground_learning_idxs = {self._learning_nusc_idx_to_idx[nusc_id] for nusc_id in NUSCENES_GROUND_LABELS}

        # --- static/dynamic in nuscenes space ---
        self._dynamic_nusc_idxs = set(NUSCENES_DYNAMIC_LABELS)
        self._static_nusc_idxs = set(NUSCENES_STATIC_LABELS)

        # --- ground in nuscenes space ---
        self._ground_nusc_idxs = set(NUSCENES_GROUND_LABELS)

        # --- points per learning class ---
        self._learning_idx_to_points = {
            l_idx: self._nusc_label_to_points_per_class[self._learning_idx_to_label[l_idx]]
            for l_idx in self._learning_idx_to_label
        }
        self._learning_idx_to_points_pseudo = {
            l_idx: self._nusc_label_to_points_per_class_pseudo[self._learning_idx_to_label[l_idx]]
            for l_idx in self._learning_idx_to_label
        }

    @property
    def free_idx(self):
        return self._free_idx

    @property
    def noise_idx(self):
        return self._noise_idx

    # --- Conversions ---
    def learning_to_dynamic(self, arr):
        return self._isin(arr, self._dynamic_learning_idxs)

    def learning_to_static(self, arr):
        return self._isin(arr, self._static_learning_idxs)

    def nusc_to_dynamic(self, arr):
        return self._isin(arr, self._dynamic_nusc_idxs)

    def nusc_to_static(self, arr):
        return self._isin(arr, self._static_nusc_idxs)

    def nusc_to_learning(self, arr):
        # The mapping depends on learning_classes provided during initialization of LabelMapper
        return self._apply_lut(arr, self._nusc_to_learning_lut)

    def lidarseg_to_learning(self, arr):
        return self._apply_lut(arr, self._lidarseg_to_learning_lut)

    def nusc_to_label(self, arr):
        return [self._nusc_idx_to_label[c] for c in np.array(arr)]

    def learning_to_color(self, arr):
        return self._apply_lut(arr, self._learning_to_color_lut)

    def learning_to_binary_target(self, arr):
        return arr != self._free_idx

    def learning_to_free(self, arr):
        return arr == self._free_idx

    def learning_to_noise(self, arr):
        return arr == self._noise_idx

    def learning_to_num_points(self, arr):
        """Map learning indices to number of points per class."""
        if isinstance(arr, torch.Tensor):
            lut_torch = torch.as_tensor(
                [self._learning_idx_to_points[i] for i in range(len(self._learning_idx_to_points))], device=arr.device
            )
            return lut_torch[arr.long()]
        elif isinstance(arr, np.ndarray):
            lut = np.array([self._learning_idx_to_points[i] for i in range(len(self._learning_idx_to_points))])
            return lut[arr]
        elif isinstance(arr, list | tuple):
            return [self._learning_idx_to_points[self._learning_label_to_idx[i]] for i in arr]
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor or list/tuple")

    def learning_to_num_points_pseudo(self, arr):
        """Map learning indices to number of points per class."""
        if isinstance(arr, torch.Tensor):
            lut_torch = torch.as_tensor(
                [self._learning_idx_to_points_pseudo[i] for i in range(len(self._learning_idx_to_points_pseudo))],
                device=arr.device,
            )
            return lut_torch[arr.long()]
        elif isinstance(arr, np.ndarray):
            lut = np.array(
                [self._learning_idx_to_points_pseudo[i] for i in range(len(self._learning_idx_to_points_pseudo))]
            )
            return lut[arr]
        elif isinstance(arr, list | tuple):
            return [self._learning_idx_to_points_pseudo[self._learning_label_to_idx[i]] for i in arr]
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor or list/tuple")

    def learning_label_to_idx(self, arr):
        if isinstance(arr, str):
            return self._learning_label_to_idx[arr]
        elif isinstance(arr, list | tuple | np.ndarray | torch.Tensor):
            return [self._learning_label_to_idx[i] for i in arr]
        else:
            raise TypeError("Input must be int, list/tuple, np.ndarray, or torch.Tensor")

    # --- helpers for NumPy / Torch ---
    @staticmethod
    def _apply_lut(arr, lut):
        if isinstance(arr, torch.Tensor):
            lut_torch = torch.as_tensor(lut, device=arr.device)
            return lut_torch[arr.long()]
        elif isinstance(arr, np.ndarray):
            return lut[arr]
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor")

    @staticmethod
    def _isin(arr, values):
        if isinstance(arr, torch.Tensor):
            return torch.isin(arr, torch.tensor(list(values), device=arr.device, dtype=arr.dtype))
        elif isinstance(arr, np.ndarray):
            return np.isin(arr, list(values))
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor")

    @staticmethod
    def _make_lut(mapping_dict, default=0):
        size = max(mapping_dict.keys()) + 1
        lut = np.full(size, default, dtype=np.int64)
        for k, v in mapping_dict.items():
            lut[k] = v
        return lut

    @staticmethod
    def _make_color_lut(mapping_dict, default=(0, 0, 0)):
        size = max(mapping_dict.keys()) + 1
        lut = np.tile(default, (size, 1)).astype(np.uint8)
        for k, v in mapping_dict.items():
            lut[k] = v
        return lut

    # --- Conversions to ZOD / AV2 ---
    def nusc_to_zod(self, arr):
        """Convert NuScenes indices or labels to ZOD labels."""
        if isinstance(arr, (list, tuple, np.ndarray, torch.Tensor)):
            labels = self.nusc_to_label(arr) if not isinstance(arr, (list, tuple)) else arr
            return [self._nusc_label_to_zod[lbl] for lbl in labels]
        elif isinstance(arr, int):
            return self._nusc_label_to_zod[self._nusc_idx_to_label[arr]]
        elif isinstance(arr, str):
            return self._nusc_label_to_zod[arr]
        else:
            raise TypeError("Input must be int, str, list, np.ndarray, or torch.Tensor")

    def nusc_to_av2(self, arr):
        """Convert NuScenes indices or labels to AV2 labels."""
        if isinstance(arr, (list, tuple, np.ndarray, torch.Tensor)):
            labels = self.nusc_to_label(arr) if not isinstance(arr, (list, tuple)) else arr
            return [self._nusc_label_to_av2[lbl] for lbl in labels]
        elif isinstance(arr, int):
            return self._nusc_label_to_av2[self._nusc_idx_to_label[arr]]
        elif isinstance(arr, str):
            return self._nusc_label_to_av2[arr]
        else:
            raise TypeError("Input must be int, str, list, np.ndarray, or torch.Tensor")
