#!/usr/bin/env python3
"""Export a dense inference checkpoint and a genuinely packed K-means artifact."""
import argparse, json, os, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
from qwen_asr import Qwen3ASRModel
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))
from kmeans_quantizer import ScalarKMeansQuantizer, pack_indices, unpack_indices


def directory_size(path): return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def reconstruct_packed(manifest, centroids, non_quantized, index_bytes):
    result = dict(non_quantized)
    for item in manifest["tensors"]:
        if not item["quantized"]: continue
        raw = index_bytes[item["packed_index_offset"]:item["packed_index_offset"] + item["packed_index_byte_count"]]
        idx = unpack_indices(raw, item["bit_width"], item["valid_index_count"])
        result[item["parameter_name"]] = centroids[item["centroid_location"]][idx].reshape(item["shape"])
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--exp_dir", required=True); a = p.parse_args()
    exp = Path(a.exp_dir); source = exp / "checkpoint-best"; dense = exp / "quantized-dense"; packed = exp / "quantized-packed"
    conf = json.loads((exp / "train_conf.json").read_text()); qconf = conf[1]["quantization"]
    wrapper = Qwen3ASRModel.from_pretrained(str(source), dtype=torch.float32, device_map="cpu")
    quantizer = ScalarKMeansQuantizer(wrapper.model, qconf["bit_width"], qconf["include_patterns"], qconf["exclude_patterns"])
    quantizer.update_and_project(False, False); dense.mkdir(parents=True, exist_ok=True)
    wrapper.model.save_pretrained(dense, safe_serialization=True); wrapper.processor.save_pretrained(dense)
    for name in ("prompt.txt", "train_conf.json"):
        src = source / name if (source / name).exists() else exp / name
        if src.exists(): shutil.copy2(src, dense / name)
    packed.mkdir(parents=True, exist_ok=True); state = wrapper.model.state_dict(); centroids = {}; nonq = {}; blob = bytearray(); tensors = []
    for name, value in state.items():
        if name in quantizer.tensors:
            qt = quantizer.tensors[name]; key = name + ".centroids"; centroids[key] = qt.centroids
            data = pack_indices(qt.assignments, qconf["bit_width"]); offset = len(blob); blob.extend(data)
            tensors.append({"parameter_name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                "bit_width": qconf["bit_width"], "centroid_location": key, "packed_index_offset": offset,
                "packed_index_byte_count": len(data), "valid_index_count": value.numel(), "quantized": True})
        else:
            nonq[name] = value.detach().cpu(); tensors.append({"parameter_name": name, "shape": list(value.shape),
                "dtype": str(value.dtype), "bit_width": None, "centroid_location": None,
                "packed_index_offset": None, "packed_index_byte_count": 0,
                "valid_index_count": value.numel(), "quantized": False})
    save_file(centroids, str(packed / "centroids.safetensors")); save_file(nonq, str(packed / "non_quantized_weights.safetensors"))
    (packed / "indices.bin").write_bytes(blob); manifest = {"format": "scalar-kmeans-v1", "tensors": tensors}
    (packed / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2))
    wrapper.processor.save_pretrained(packed)
    for name in ("config.json", "generation_config.json", "prompt.txt"):
        src = dense / name
        if src.exists(): shutil.copy2(src, packed / name)
    restored = reconstruct_packed(manifest, centroids, nonq, bytes(blob))
    for name, value in state.items():
        if not torch.equal(restored[name].to(value.dtype), value.cpu()): raise RuntimeError(f"Packed reconstruction mismatch: {name}")
    original = directory_size(source); dense_size = directory_size(dense); packed_size = directory_size(packed)
    selected = sum(state[n].numel() for n in quantizer.tensors); total = sum(v.numel() for v in state.values())
    report = {"original_checkpoint_bytes": original, "dense_quantized_bytes": dense_size,
              "packed_artifact_bytes": packed_size, "packed_compression_ratio": original / packed_size,
              "quantized_parameter_count": selected, "total_parameter_count": total,
              "quantized_parameter_percentage": 100 * selected / total, "packed_reconstruction_test": "passed"}
    (exp / "compression_report.json").write_text(json.dumps(report, indent=2))
    (exp / "compression_report.txt").write_text("\n".join(f"{k}: {v}" for k, v in report.items()) + "\n")

if __name__ == "__main__": main()
