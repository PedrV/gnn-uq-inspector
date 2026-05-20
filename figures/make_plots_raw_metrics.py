import os
import subprocess
import argparse
import re

from pathlib import Path

DATASET_TYPE = {
    "regression": ("pems", "artnetviews", "chameleon", "gapsmallqm9"),
    "classification": ("cora", "citeseer", "tolokers2")
}

oscwd = Path(os.getcwd())

EXPERIMENTS_STORAGE = {
    "pems": "pems/original/gcn/tristan_2026-04-02_15-53-04",
    "artnetviews": "artnetviews/original/megagcn/phoenix_2026-04-02_12-45-00",
    "chameleon": "chameleon/binary_features_logarithmic_target/megagat/zeus_2026-04-02_15-42-51",
    "gapsmallqm9": "gapsmallqm9/original/gcngraph/gareth_2026-04-02_15-42-55",
    "cora": "cora/original/gcn/perceval_2026-03-02_15-34-05",
    "citeseer": "citeseer/original/gcn/arthur_2026-03-02_18-27-00",
    "tolokers2": "tolokers2/original/megagat/mordred_2026-03-27_19-04-29"
}
EXPERIMENTS_STORAGE_BASE = f"{oscwd.parent.parent / "do_figures"}/outputs"


EXPERIMENTS_STORAGE_OOD = {
    "artnetviews": "artnetviews/shift_original/megagcn/hector_2026-04-30_16-03-43",
    "chameleon": "chameleon/shift_binary_features_logarithmic_target/megagat/bagdemagus_2026-04-28_21-50-27",
    "tolokers2": "tolokers2/shift_original/megagat/mordred_2026-05-01_08-56-24"
}

def build_dir_string(dataset, final_dir, is_ood=False):
    base_container_dir = "/workspace/existing_run"
    try:
        if is_ood:
            full_path_dir = os.path.join(base_container_dir, EXPERIMENTS_STORAGE_OOD[dataset], final_dir)
        else:
            full_path_dir = os.path.join(base_container_dir, EXPERIMENTS_STORAGE[dataset], final_dir)
    except KeyError:
        return ""

    return full_path_dir

datasets_nll = {
    "cora": {
        "dataset_dirs": [f'{build_dir_string("cora", "")}'],
        "point_metric": "Accuracy Test",
        "uq_compare_metric": "NLL",
        "models": ["stats_uq_total", "stats_individual"],
        "models_size": ["5", "5"],
    }, 
    "citeseer": {
        "dataset_dirs": [f'{build_dir_string("citeseer", "")}'],
        "point_metric": "Accuracy Test",
        "uq_compare_metric": "NLL",
        "models": ["stats_uq_total", "stats_individual"],
        "models_size": ["5", "5"],
    }, 
    "tolokers2": {
        "dataset_dirs": [f'{build_dir_string("tolokers2", "")}'],
        "point_metric": "Accuracy Test",
        "uq_compare_metric": "NLL",
        "models": ["stats_uq_total", "stats_individual"],
        "models_size": ["5", "5"],
    },
    "artnetviews": {
        "dataset_dirs": [f'{build_dir_string("artnetviews", "")}'],
        "point_metric": "RMSE Test",
        "uq_compare_metric": "Diagonal NLL",
        "models": ["stats_uq_total_copy", "stats_uq_total", "stats_new_baseline", "stats_uq_aleatoric"],
        "models_size": ["1", "5", "5", "5"],
    }, 
    "chameleon": {
        "dataset_dirs": [f'{build_dir_string("chameleon", "")}'],
        "point_metric": "RMSE Test",
        "models": ["stats_uq_total_copy", "stats_uq_total", "stats_new_baseline", "stats_uq_aleatoric"],
        "models_size": ["1", "5", "5", "5"],
    },
    "gapsmallqm9": {
        "dataset_dirs": [f'{build_dir_string("gapsmallqm9", "")}'],
        "point_metric": "RMSE Test",
        "uq_compare_metric": "Diagonal NLL",
        "models": ["stats_uq_total_copy", "stats_uq_total", "stats_new_baseline", "stats_uq_aleatoric"],
        "models_size": ["1", "5", "5", "5"],
    },
}

datasets_point_ood = {
    "tolokers2": {
        "dataset_dirs": [f'{build_dir_string("tolokers2", "", is_ood=True)}'],
        "uq_compare_metric": "AP Test",
        "models": ["stats_uq_total_ood_copy", "stats_uq_total_ood"],
        "models_size": ["1", "5"],
    },
    "artnetviews": {
        "dataset_dirs": [f'{build_dir_string("artnetviews", "", is_ood=True)}'],
        "models": ["stats_uq_total_ood_copy", "stats_uq_total_ood"],
        "uq_compare_metric": "R2 Test",
        "models_size": ["1", "5"],
    }, 
    "chameleon": {
        "dataset_dirs": [f'{build_dir_string("chameleon", "", is_ood=True)}'],
        "models": ["stats_uq_total_ood_copy", "stats_uq_total_ood"],
        "uq_compare_metric": "RMSE Test",
        "models_size": ["1", "5"],
    },
}


datasets_soup = {
    "cora": {
        "dataset_dirs": [f'{build_dir_string("cora", "")}'],
        "point_metric": "Accuracy Test",
        "uq_compare_metric": "NLL",
        "models": ["stats_uq_total_for_soup", "soup"],
        "models_size": ["1", "soup"],
    }, 
    "citeseer": {
        "dataset_dirs": [f'{build_dir_string("citeseer", "")}'],
        "point_metric": "Accuracy Test",
        "uq_compare_metric": "NLL",
        "models": ["stats_uq_total_for_soup", "soup"],
        "models_size": ["1", "soup"],
    }, 
    "tolokers2": {
        "dataset_dirs": [f'{build_dir_string("tolokers2", "")}'],
        "point_metric": "Accuracy Test",
        "uq_compare_metric": "NLL",
        "models": ["stats_uq_total_for_soup", "soup"],
        "models_size": ["1", "soup"],
    },
    "artnetviews": {
        "dataset_dirs": [f'{build_dir_string("artnetviews", "")}'],
        "point_metric": "RMSE Test",
        "uq_compare_metric": "Diagonal NLL",
        "models": ["stats_uq_total_for_soup", "soup"],
        "models_size": ["1", "soup"],
    }, 
    "chameleon": {
        "dataset_dirs": [f'{build_dir_string("chameleon", "")}'],
        "point_metric": "RMSE Test",
        "uq_compare_metric": "Diagonal NLL",
        "models": ["stats_uq_total_for_soup", "soup"],
        "models_size": ["1", "soup"],
    },
    "gapsmallqm9": {
        "dataset_dirs": [f'{build_dir_string("gapsmallqm9", "")}'],
        "point_metric": "RMSE Test",
        "uq_compare_metric": "Diagonal NLL",
        "models": ["stats_uq_total_for_soup", "soup"],
        "models_size": ["1", "soup"],
    },
}  # stats_uq_total_for_soup = stats_uq_total + stats_extra (if it exists)

SCRIPT_PATH = f"{oscwd.parent}/run_in_container.fish"

def run(datasets, dataset_type, type_plot, misc_type):
    SOUP = misc_type == "soup"
    OOD = misc_type == "ood"
    if not SOUP and not OOD:
        if dataset_type == "classification":
            models_label = ["stats_uq_total", "DE", "stats_individual", "Expected Individual"]
        else:
            models_label = ["stats_uq_total_copy", "Single", "stats_uq_total", "DE", "stats_new_baseline", "DE-R", "stats_uq_aleatoric", "DE-A"]
        datasets_specs = datasets_nll
    elif SOUP:
        models_label = ["stats_uq_total_for_soup", "Single", "soup", "Uniform Soup"]
        datasets_specs = datasets_soup
    else:
        models_label = ["stats_uq_total_ood_copy", "Single", "stats_uq_total_ood", "DE"]
        datasets_specs = datasets_point_ood

    options = {}
    if not SOUP and not OOD:
        options["out"] = f"{type_plot}_baselines_{dataset_type}"     
    elif SOUP:
        options["out"] =  f"{type_plot}_soup_{dataset_type}"
    else:
        options["out"] =  f"{type_plot}_ood_{dataset_type}"

    if type_plot == "nll":
        options["metric"] = "NLL" if dataset_type == "classification" else "Diagonal NLL"
    else:
        if not OOD:
            options["metric"] = "Accuracy Test" if dataset_type == "classification" else "RMSE Test"
        elif OOD:
            options["metric"] = ["AP Test", "R2 Test"]
    cmd = [
        SCRIPT_PATH, "cpu", "1", EXPERIMENTS_STORAGE_BASE, "do", "python", "figures/plot_raw_metrics.py",
        "--out",        options["out"]+".pdf",
    ]

    cmd.append("--metric")
    if not OOD:
       cmd.append(options["metric"])
    else:
        for _m in options["metric"]:
            cmd.append(_m)

    cmd.append("--dataset_dirs")
    for name, cfg in datasets_specs.items():
        if name not in datasets:
            continue
        for run_dir in cfg["dataset_dirs"]:
            cmd.append(run_dir)

    cmd.append("--dataset_labels")
    for name, cfg in datasets_specs.items():
        if name not in datasets:
            continue
        if name == "gapsmallqm9":
            cmd.append("QM9")
        else:
            cmd.append(name.title())

    for name, cfg in datasets_specs.items():
        if name not in datasets:
            continue
        cmd.append("--models")
        for m in cfg["models"]:
            cmd.append(m)

    for name, cfg in datasets_specs.items():
        if name not in datasets:
            continue
        cmd.append("--models_size")
        for m in cfg["models_size"]:
            cmd.append(m) 

    cmd.append("--models_label")
    for ml in models_label:
        cmd.append(ml)
    
    if dataset_type == "classification" and not SOUP:
        cmd.append("--draw_arrows")
        cmd.append("stats_uq_total")
        cmd.append("stats_individual")
    elif SOUP:
        cmd.append("--draw_arrows")
        cmd.append("stats_uq_total_for_soup")
        cmd.append("soup")

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
    parser.add_argument(
        "--type",
        required=True,
        choices=["nll", "point"],
        help="Do NLL or Point prediction",
    )
    parser.add_argument(
        "--misc_type",
        required=True,
        choices=["soup", "normal", "ood"],
        help="Soup or Normal",
    )
    args = parser.parse_args()

    dataset_type = ""
    for dt in args.datasets:
        if dt in DATASET_TYPE["classification"] and (dataset_type == "classification" or dataset_type == ""):
            dataset_type = "classification"
        elif dt in DATASET_TYPE["regression"] and (dataset_type == "regression" or dataset_type == ""):
            dataset_type = "regression"
        else:
            print("Only for OOD!")
            dataset_type = "mixed"

    run(tuple(args.datasets), dataset_type, args.type, args.misc_type)
