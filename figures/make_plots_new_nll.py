import os
import subprocess
import argparse
import re

from pathlib import Path

import numpy as np

OOD = False

DATASET_TYPE = {
    "regression": ("pems", "artnetviews", "chameleon", "gapsmallqm9"),
    "classification": ("cora", "citeseer", "tolokers2")
}

oscwd = Path(os.getcwd())

DATASET_SIZE = {
    "pems": 75,
    "cora": 1, # 1000,
    "citeseer": 1, # 1000,
    "chameleon": 456,
    "gapsmallqm9": 818,
    "artnetviews": 12791,
    "tolokers2": 1, # 2974,
}

EXPERIMENTS_STORAGE = {
    "pems": "pems/original/gcn/tristan_2026-04-02_15-53-04",
    "artnetviews": "artnetviews/original/megagcn/phoenix_2026-04-02_12-45-00",
    "chameleon": "chameleon/binary_features_logarithmic_target/megagat/zeus_2026-04-02_15-42-51",
    "gapsmallqm9": "gapsmallqm9/original/gcngraph/gareth_2026-04-02_15-42-55",
    "cora": "cora/original/gcn/perceval_2026-03-02_15-34-05",
    "citeseer": "citeseer/original/gcn/arthur_2026-03-02_18-27-00",
    "tolokers2": "tolokers2/original/megagat/mordred_2026-03-27_19-04-29"
}

EXPERIMENTS_STORAGE_OOD = {
    "artnetviews": "artnetviews/shift_original/megagcn/hector_2026-04-30_16-03-43",
    "chameleon": "chameleon/shift_binary_features_logarithmic_target/megagat/bagdemagus_2026-04-28_21-50-27",
    "tolokers2": "tolokers2/shift_original/megagat/mordred_2026-05-01_08-56-24"
}

EXPERIMENTS_STORAGE_BASE = f"{oscwd.parent.parent / "do_figures"}/outputs"

def build_dir_string(dataset, final_dir):
    base_container_dir = "/workspace/existing_run"
    try:
        if OOD:
            full_path_dir = os.path.join(base_container_dir, EXPERIMENTS_STORAGE_OOD[dataset], final_dir)
        else:
            full_path_dir = os.path.join(base_container_dir, EXPERIMENTS_STORAGE[dataset], final_dir)
    except KeyError:
        return ""

    return full_path_dir

datasets_nll = {
    "cora": {
        "dataset_dirs": [f'{build_dir_string("cora", "stats_uq_total")}'],
        "uq_compare_metric": "NLL",
    }, 
    "citeseer": {
        "dataset_dirs": [f'{build_dir_string("citeseer", "stats_uq_total")}'],
        "uq_compare_metric": "NLL",
    }, 
    "tolokers2": {
        "dataset_dirs": [f'{build_dir_string("tolokers2", "stats_uq_total")}'],
        "uq_compare_metric": "NLL",
    },
    "pems": {
        "dataset_dirs": [f'{build_dir_string("pems", "stats_uq_total")}'],
        "uq_compare_metric": "Diagonal NLL",
    },
    "artnetviews": {
        "dataset_dirs": [f'{build_dir_string("artnetviews", "stats_uq_total")}'],
        "uq_compare_metric": "Diagonal NLL",
    }, 
    "chameleon": {
        "dataset_dirs": [f'{build_dir_string("chameleon", "stats_uq_total")}'],
        "uq_compare_metric": "Diagonal NLL",
    },
    "gapsmallqm9": {
        "dataset_dirs": [f'{build_dir_string("gapsmallqm9", "stats_uq_total")}'],
        "uq_compare_metric": "Diagonal NLL",
    },
}

datasets_nll_ood = {
    "tolokers2": {
        "dataset_dirs": [f'{build_dir_string("tolokers2", "stats_uq_total_ood")}'],
        "uq_compare_metric": "NLL",
    },
    "artnetviews": {
        "dataset_dirs": [f'{build_dir_string("artnetviews", "stats_uq_total_ood")}'],
        "uq_compare_metric": "Diagonal NLL",
    }, 
    "chameleon": {
        "dataset_dirs": [f'{build_dir_string("chameleon", "stats_uq_total_ood")}'],
        "uq_compare_metric": "Diagonal NLL",
    },
}

baselines = {
    # 1,2,3: https://arxiv.org/abs/1612.01474; 4: https://arxiv.org/pdf/2006.10108
    "classification_nll": [f"ImageNet, Inception={np.exp(0.17)}", f"MNIST, MLP={np.exp(0.08)}", f"SVHN, VGG={np.exp(0.15)}", f"CIFAR10, ResNet={np.exp(0.208)}"],
    # 1: https://arxiv.org/abs/2403.05600, 2: https://arxiv.org/abs/1612.01474
    "regression_nll": [f"NYUdepthV2, UNet={np.exp(0.1920)}"],
    "mixed_nll": []
}

SCRIPT_PATH = f"{oscwd.parent}/run_in_container.fish"

def run(datasets, dataset_type):
    options =  {
        "labels": ["10", "5"],
        "out": f"new_style_{dataset_type}",
        "ensemble_position": ["4", "5"],
        "ensemble_base": "0",
    }
    
    options["baselines"] = baselines[dataset_type+"_nll"]
    options["bottom_annot"] = ""
    
    if dataset_type != "mixed":
        options["metric"] = "NLL" if dataset_type == "classification" else "Diagonal NLL"
    else:
        options["metric"] = []
        for _data in datasets:
            is_classification = _data in DATASET_TYPE["classification"]
            options["metric"].append("NLL" if is_classification else "Diagonal NLL")

    cmd = [
        SCRIPT_PATH, "cpu", "1", EXPERIMENTS_STORAGE_BASE, "do", "python", "figures/plotter_new_nll.py",
        "--out",        options["out"]+f"_new_nll_{OOD}.pdf",
        "--ensemble_base", options["ensemble_base"],
        "--bottom_annot", options["bottom_annot"]
    ]

    dataset_to_use = datasets_nll_ood if OOD else datasets_nll

    cmd.append("--metric")
    if dataset_type != "mixed":
       cmd.append(options["metric"])
    else:
        for _m in options["metric"]:
            cmd.append(_m)

    cmd.append("--ensemble_position")
    for pos in options["ensemble_position"]:
        cmd.append(pos)

    if "baselines" in options and not OOD:
        cmd.append("--baselines")
        for bas in options["baselines"]:
            cmd.append(bas)
    
    cmd.append("--dataset_dirs")
    for name, cfg in dataset_to_use.items():
        if name not in datasets:
            continue
        for run_dir in cfg["dataset_dirs"]:
            cmd.append(run_dir)

    cmd.append("--dataset_labels")
    for name, cfg in dataset_to_use.items():
        if name not in datasets:
            continue
        if name == "gapsmallqm9":
            cmd.append("QM9")
        else:
            cmd.append(name.title())

    cmd.append("--dataset_sizes")
    for name, cfg in dataset_to_use.items():
        if name not in datasets:
            continue
        cmd.append(f"{name}={DATASET_SIZE[name]}")
    
    cmd.append("--bundle")
    cmd.append("iclr2024")
    
    print(cmd)
    subprocess.run(cmd, cwd=oscwd.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotter orchestrator."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Datasets to plot metrics over.",
    )
    args = parser.parse_args()

    dataset_type = ""
    for dt in args.datasets:
        if dt in DATASET_TYPE["classification"] and (dataset_type == "classification" or dataset_type == ""):
            dataset_type = "classification"
        elif dt in DATASET_TYPE["regression"] and (dataset_type == "regression" or dataset_type == ""):
            dataset_type = "regression"
        else:
            dataset_type = "mixed"

    run(tuple(args.datasets), dataset_type)
