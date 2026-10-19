"""Physical Celsius observations; clean targets never enter model arguments."""
import hashlib
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Sampler
from eo_data import open_dataset
from .config import COUNTS, PREPARED, digest
from .corruptions import corrupt


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") & ((1 << 63) - 1)


class LSTDataset(Dataset):
    def __init__(self, config, split, evaluation_profile=None, view=0):
        self.config, self.split, self.view = config, split, view
        if split not in COUNTS or config["data"]["prepared"] != PREPARED:
            raise ValueError("Explicit frozen HLP v2 required")
        root = Path(config["data"]["root"])
        self.source = open_dataset("prithvi", "lst", split, root=root, prepared=root / PREPARED)
        self.record_ids = np.asarray(self.source.indices[:, 0])
        if len(self.source) != COUNTS[split] or len(np.unique(self.record_ids)) != len(self.source):
            raise ValueError("Incomplete/duplicate split")
        if self.source.stats["fit_split"] != "train" or self.source.stats["samples"] != COUNTS["train"]:
            raise ValueError("Normalization is not the remaining v2 train statistics")
        self.profile = evaluation_profile
        if split != "train" and evaluation_profile is None:
            raise ValueError("Evaluation condition must be explicit")
        self.fingerprint = {"prepared": PREPARED, "source": self.source.fingerprint,
            "stats": self.source.stats, "split": split, "samples": len(self),
            "member_sha256": hashlib.sha256(self.record_ids.astype("<i8").tobytes()).hexdigest(),
            "corruption_sha256": digest(evaluation_profile if evaluation_profile else config["corruption"]),
            "unit": "degree_Celsius", "mean_source": config["task"]["mean_source"]}

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) else (0, int(item))
        sample = self.source[index]
        valid = np.asarray(sample["reference_mask"], dtype=bool)
        temperature = np.asarray(self.source.denormalize(sample["image"]) / 100., dtype=np.float32)
        if list(temperature.shape) != self.config["data"]["shape"]:
            raise ValueError("Unregistered image shape")
        if not valid.any() or not np.isfinite(temperature[valid]).all():
            raise ValueError("Invalid reference temperature")
        temperature = np.where(valid, temperature, np.float32(0)).astype(np.float32)
        sid = sample["sample_id"]
        selected = self.profile
        if selected is None:
            profiles = self.config["corruption"]["train_profiles"]
            number = stable_seed("profile", self.config["corruption"]["profile_selection_seed"], epoch, sid)
            selected = profiles[number % len(profiles)]
        noisy = corrupt(temperature, valid, selected, sample_id=sid, split=self.split, epoch=epoch, view=self.view)
        observed = noisy["observed_valid"]
        if self.config["task"]["mean_source"] == "oracle":
            mean = np.float32(temperature[valid].mean(dtype=np.float64))
        else:
            # Includes sensor perturbation: never borrow clean observed-target values.
            mean = np.float32(noisy["observation"][observed].mean(dtype=np.float64))
        center = mean if self.config["task"]["field"] == "residual" else np.float32(self.config["task"]["absolute_center_celsius"])
        input_field = np.where(observed, noisy["observation"] - center, np.float32(0)).astype(np.float32)
        target_field = np.where(valid, temperature - center, np.float32(0)).astype(np.float32)
        return {"sample_id": sid, "tile_id": sample["tile_id"], "record_id": int(self.record_ids[index]),
                "temperature": temperature, "reference_valid": valid, "observation": noisy["observation"],
                "observed_valid": observed, "missing_mask": noisy["missing_mask"],
                "affected_mask": noisy["affected_mask"], "patch_missing": noisy["patch_missing"],
                "input_field": input_field, "target_field": target_field,
                "mean_celsius": mean, "center_celsius": center, "noise_seed": noisy["seed"],
                "profile": noisy["profile"], "view": self.view}

    def close(self):
        self.source.close()


class EpochSampler(Sampler):
    def __init__(self, length, seed, epoch, consumed=0):
        if length < 1 or not 0 <= consumed <= length or epoch < 0:
            raise ValueError("Invalid epoch sampler cursor")
        self.length, self.seed, self.epoch, self.consumed = length, seed, epoch, consumed

    def __iter__(self):
        order = np.random.default_rng(stable_seed("order", self.seed, self.epoch)).permutation(self.length)
        for i in order[self.consumed:]:
            yield self.epoch, int(i)

    def __len__(self):
        return self.length - self.consumed


def update_batches(iterator, accumulation):
    while True:
        batches = []
        for _ in range(accumulation):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        if not batches:
            break
        yield batches
