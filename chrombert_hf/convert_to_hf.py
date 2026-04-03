import argparse
import os
import shutil
import torch

from configuration_chrombert import ChromBERTConfig
from modeling_chrombert import ChromBERTModel


def _normalize_state_dict(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    num_prefixed = len([key for key in state_dict if key.startswith("model.")])
    if state_dict and num_prefixed / len(state_dict) > 0.9:
        state_dict = {key[6:]: value for key, value in state_dict.items() if key.startswith("model.")}

    return state_dict


def convert_checkpoint(
    ckpt_path,
    output_dir,
    genome="hg38",
    dropout=0.1,
    dtype_str="bfloat16",
):
    os.makedirs(output_dir, exist_ok=True)

    config = ChromBERTConfig(
        genome=genome,
        dropout=dropout,
        dtype_str=dtype_str,
        architectures=["ChromBERTModel"],
        auto_map={
            "AutoConfig": "configuration_chrombert.ChromBERTConfig",
            "AutoModel": "modeling_chrombert.ChromBERTModel",
        },
    )

    model = ChromBERTModel(config)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = _normalize_state_dict(state_dict)
    model.chrombert.load_state_dict(state_dict)
    model.save_pretrained(output_dir, safe_serialization=False)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(os.path.join(repo_root, "configuration_chrombert.py"), output_dir)
    shutil.copy2(os.path.join(repo_root, "modeling_chrombert.py"), output_dir)
    shutil.copytree(
        os.path.join(repo_root, "chrombert"),
        os.path.join(output_dir, "chrombert"),
        dirs_exist_ok=True,
    )

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert a ChromBERT checkpoint to a HF-style directory.")
    parser.add_argument("--ckpt-path", required=True, help="Path to the original ChromBERT checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the HF model files.")
    parser.add_argument("--genome", default="hg38", help="Genome for the checkpoint, default: hg38.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value stored in config.json.")
    parser.add_argument("--dtype-str", default="bfloat16", help="Torch dtype string stored in config.json.")
    args = parser.parse_args()

    convert_checkpoint(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        genome=args.genome,
        dropout=args.dropout,
        dtype_str=args.dtype_str,
    )


if __name__ == "__main__":
    main()
