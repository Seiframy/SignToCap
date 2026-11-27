"""Batch-one Uni-Sign adapter for pinned upstream code."""

import sys
from time import perf_counter
from types import SimpleNamespace


class Translator:
    def __init__(self, upstream, mt5, checkpoint, *, device="cuda"):
        import numpy as np
        import torch

        sys.path.insert(0, str(upstream.resolve()))
        import models
        from datasets import load_part_kp

        self.np = np
        self.torch = torch
        self.load_part_kp = load_part_kp
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "cuda":
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("The benchmark requires a CUDA GPU with BF16 support")
            torch.cuda.reset_peak_memory_stats()
        elif device != "cpu":
            raise ValueError("Expected CUDA or CPU")
        models.mt5_path = str(mt5.resolve())
        args = SimpleNamespace(dataset="How2Sign", hidden_dim=256, rgb_support=False, label_smoothing=0.0)
        model = models.Uni_Sign(args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        del state
        self.model = model.eval().to(device=device, dtype=self.dtype)

    def source(self, keypoints, scores):
        torch = self.torch
        if not 0 < len(keypoints) <= 256 or len(keypoints) != len(scores):
            raise ValueError("Expected one to 256 paired pose frames")
        parts = self.load_part_kp(keypoints, scores, force_ok=True)
        source = {part: value.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                  for part, value in parts.items()}
        source["attention_mask"] = torch.ones((1, len(keypoints)), device=self.device, dtype=torch.long)
        return source

    def load_pose_adapter(self, path, checkpoint):
        import hashlib

        with checkpoint.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("base_sha256") != digest:
            raise ValueError("Pose adapter belongs to a different base checkpoint")
        self.model.pose_proj.load_state_dict(payload["pose_proj"], strict=True)

    def translate(self, keypoints, scores, *, beams=1, tokens=48, source_only=True):
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        with torch.inference_mode():
            source = self.source(keypoints, scores)
            prepared = perf_counter()
            embeddings = self.model(source, {"gt_sentence": [""]}, source_only=source_only)
            if self.device == "cuda":
                torch.cuda.synchronize()
            encoded = perf_counter()
            ids = self.model.generate(embeddings, max_new_tokens=tokens, num_beams=beams)
            token_ids = ids[0].tolist()
            text = self.model.mt5_tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        finished = perf_counter()
        return text, token_ids, {
            "prep_ms": (prepared - started) * 1000,
            "pose_encode_ms": (encoded - prepared) * 1000,
            "generation_ms": (finished - encoded) * 1000,
            "work_ms": (finished - started) * 1000,
        }

    def memory(self):
        cuda = self.torch.cuda
        return {
            "torch_peak_allocated_mib": round(cuda.max_memory_allocated() / 2**20, 1),
            "torch_peak_reserved_mib": round(cuda.max_memory_reserved() / 2**20, 1),
        }
