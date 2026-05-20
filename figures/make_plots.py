import os
import subprocess
import argparse
import re

from pathlib import Path

GRID = False

DATASET_TYPE = {
    "regression": ("pems", "artnetviews", "chameleon", "gapsmallqm9"),
    "classification": ("cora", "citeseer", "tolokers2")
}

oscwd = Path(os.getcwd())

EXPERIMENTS_STORAGE = {
    "pems": f"{oscwd.parent.parent / "do_figures"}/outputs/pems/original/gcn/tristan_2026-04-02_15-53-04",
    "artnetviews": f"{oscwd.parent.parent / "do_figures"}/outputs/artnetviews/original/megagcn/phoenix_2026-04-02_12-45-00",
    "chameleon": f"{oscwd.parent.parent / "do_figures"}/outputs/chameleon/binary_features_logarithmic_target/megagat/zeus_2026-04-02_15-42-51",
    "gapsmallqm9": f"{oscwd.parent.parent / "do_figures"}/outputs/gapsmallqm9/original/gcngraph/gareth_2026-04-02_15-42-55",
    "cora": f"{oscwd.parent.parent / "do_figures"}/outputs/cora/original/gcn/perceval_2026-03-02_15-34-05",
    "citeseer": f"{oscwd.parent.parent / "do_figures"}/outputs/citeseer/original/gcn/arthur_2026-03-02_18-27-00",
    "tolokers2": f"{oscwd.parent.parent / "do_figures"}/outputs/tolokers2/original/megagat/mordred_2026-03-27_19-04-29"
}
EXPERIMENTS_STORAGE_GRID = f"{oscwd.parent.parent / "do_figures"}/outputs"

def build_dir_string(dataset, final_dir):
    base_container_dir = "/workspace/existing_run"
    if GRID:
        _last_part = re.findall(EXPERIMENTS_STORAGE_GRID + r'(.+)', EXPERIMENTS_STORAGE[dataset])[0]
    else:
        _last_part = ""
    full_path_dir = os.path.join(base_container_dir, _last_part, final_dir)
    return full_path_dir

datasets_point_estimation = {
    "artnetviews": {
        "run_dir": [f'{build_dir_string("artnetviews", "stats_uq_total")}'],
        "metric": "R2 Test",
        "out": "artnetviews_r2.pdf",
        "labels": "R2",
        "baselines": [f"Trivial={0}"]
    },
    "pems": {
        "run_dir": [f'{build_dir_string("pems", "stats_uq_total")}'],
        "metric": "RMSE Test",
        "out": "pems_rmse.pdf",
        "labels": "RMSE",
        "baselines": [f"Trivial={16.4320}"]
    },
    "chameleon": {
        "run_dir": [f'{build_dir_string("chameleon", "stats_uq_total")}'],
        "metric": "RMSE Test",
        "out": "chameleon_rmse.pdf",
        "labels": "RMSE",
        "baselines": [f"Trivial={2.1600}"]
    },
    "gapsmallqm9": {
        "run_dir": [f'{build_dir_string("gapsmallqm9", "stats_uq_total")}'],
        "metric": "RMSE Test",
        "out": "gapsmallqm9_rmse.pdf",
        "labels": "RMSE",
        "baselines": [f"Trivial={1.2681}"]
    },
    "tolokers2": {
        "run_dir": [f'{build_dir_string("tolokers2", "stats_uq_total")}'],
        "metric": "AP Test",
        "out": "tolokers2_ap.pdf",
        "labels": "Average Precision",
        "baselines": [f"Trivial={0.2182}"]
    },
    "cora": {
        "run_dir": [f'{build_dir_string("cora", "stats_uq_total")}'],
        "metric": "Accuracy Test",
        "out": "cora_accuracy.pdf",
        "labels": "Accuracy",
        "baselines": [f"Trivial={0.13}"]
    },
    "citeseer": {
        "run_dir": [f'{build_dir_string("citeseer", "stats_uq_total")}'],
        "metric": "Accuracy Test",
        "out": "citeseer_accuracy.pdf",
        "labels": "Accuracy",
        "baselines": [f"Trivial={0.0770}"]
    },
}

datasets_nll = {
    "artnetviews": {
        "run_dir": [f'{build_dir_string("artnetviews", "stats_uq_total")}'],
        "metric": "Diagonal NLL",
        "out": "artnetviews_nll.pdf",
        "labels": ["DE"],
        "decomp": [0, 3],
        "baselines": [f"Trivial={18223.6030/12791}"]
    },
    "chameleon": {
        "run_dir": [f'{build_dir_string("chameleon", "stats_uq_total")}'],
        "metric": "Diagonal NLL",
        "out": "chameleon_nll.pdf",
        "labels": ["DE"],
        "decomp": [0, 3],
        "baselines": [f"Trivial={645.2080/456}"]
    },
    "gapsmallqm9": {
        "run_dir": [f'{build_dir_string("gapsmallqm9", "stats_uq_total")}'],
        "metric": "Diagonal NLL",
        "out": "gapsmallqm9_nll.pdf",
        "labels": ["DE"],
        "decomp": [0, 2],
        "baselines": [f"Trivial={1148.1616/818}"]
    },
    "pems": {
        "run_dir": [f'{build_dir_string("pems", "stats_uq_total")}'],
        "metric": "Diagonal NLL",
        "out": "pems_nll.pdf",
        "labels": ["DE"],
        "decomp": [0, 2],
        "baselines": [f"Trivial={102.6190/75}"]
    },
    "cora": {
        "run_dir": [f'{build_dir_string("cora", "stats_uq_total")}'],
        "metric": "NLL",
        "out": "cora_nll.pdf",
        "labels": ["DE"],
        "decomp": [0, 1],
        "baselines": [f"Trivial={1.9460}"]
    }, 
    "citeseer": {
        "run_dir": [f'{build_dir_string("citeseer", "stats_uq_total")}'],
        "out": "citeseer_nll.pdf",
        "metric": "NLL",
        "labels": ["DE"],
        "decomp": [0, 1],
        "baselines": [f"Trivial={1.7920}"]
    }, 
    "tolokers2": {
        "run_dir": [f'{build_dir_string("tolokers2", "stats_uq_total")}'],
        "metric": "NLL",
        "out": "tolokers2_nll.pdf",
        "labels": ["DE"],
        "decomp": [0, 1],
        "baselines": [f"Trivial={0.5247}"]
    }, 
}

SCRIPT_PATH = f"{oscwd.parent}/run_in_container.fish"

def run(datasets, plot_type, dataset_type):
    storage = EXPERIMENTS_STORAGE
    point_dataset = datasets_point_estimation
    nll_dataset = datasets_nll

    if plot_type == "point":
        for name, cfg in point_dataset.items():
            if name not in datasets:
                continue
            cmd = [
                SCRIPT_PATH, "cpu", "1", storage[name], "do", "python", "figures/plotter.py",
                "--run_dir",    cfg["run_dir"][0],
                "--metric",     cfg["metric"],
                "--out",        cfg["out"],
                "--dataset",    name,
                "--style",      "bars",
                "--labels",     cfg["labels"]
            ]

            if "baselines" in cfg:
                cmd.append("--baselines")
                for bas in cfg["baselines"]:
                    cmd.append(bas)
            print(cmd)
            subprocess.run(cmd, cwd=oscwd.parent)
    elif plot_type == "nll":
        for name, cfg in nll_dataset.items():
            if name not in datasets:
                continue

            cmd = [
                SCRIPT_PATH, "cpu", "1", storage[name], "do", "python", "figures/plotter.py",
                "--metric",     cfg["metric"],
                "--out",        cfg["out"],
                "--dataset",    name,
                "--style",      "bars",
            ]

            cmd.append("--run_dir")
            for run_dir in cfg["run_dir"]:
                cmd.append(run_dir)

            cmd.append("--labels")
            for label in cfg["labels"]:
                cmd.append(label)
                
            if "decomp" in cfg:
                cmd.append("--decomp")
                for idx in cfg["decomp"]:
                    cmd.append(str(idx))

            if "baselines" in cfg:
                cmd.append("--baselines")
                for bas in cfg["baselines"]:
                    cmd.append(bas)
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
        choices=["point", "nll"],
        default="point",
        help="Do plots for point estimation or NLL",
    )
    args = parser.parse_args()
    dataset_type = ""
    for dt in args.datasets:
        if dt in DATASET_TYPE["classification"] and (dataset_type == "classification" or dataset_type == ""):
            dataset_type = "classification"
        elif dt in DATASET_TYPE["regression"] and (dataset_type == "regression" or dataset_type == ""):
            dataset_type = "regression"
        else:
            raise Exception("All datasets must be of the same type")

    run(tuple(args.datasets), args.type, dataset_type)
