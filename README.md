
## Dataset Setup

Follow the steps below to download and prepare the datasets:

### 1. Download the datasets
- Download the [ready_to_use_data.tar.gz](https://figshare.com/s/f1a5ef5e9aa77f5fe18a) provided.
- If you want to build the datasets from scratch you only need the raw data:
  - Delete the `processed` folder inside each dataset directory after downloading.
- Alternatively, you can download the raw datasets directly from their original sources (refer to the associated papers).

### 2. Place the datasets
- Move the `datasets` folder into the root directory of the repository:

```
	gnn-uq-inspector/
	            └── datasets/
```

### 3. Next steps
- Once the datasets are correctly placed, proceed to the next section[^1].

**Note on PEMS dataset:**
To build the dataset from scratch place [processed_pems_data.pkl](https://figshare.com/s/f1a5ef5e9aa77f5fe18a) in the root directory: ```gnn-uq-inspector/```.
To use the original raw data download the files from the original source: [PeMS Regression Benchmark (Borovitskiy)](https://github.com/vabor112/pems-regression/tree/main/pems_regression/resources).
Run `python make_pems.py`, this will generate the dataset inside the `pems/` folder.
After generation, move the `pems/` folder into: `gnn-uq-inspector/datasets/`.

# Running the experiments

 1. Install [podman](https://podman.io/docs/installation). 
 2. `chmod +x run.sh run_in_container.sh`
 3. `./run.sh gpu|cpu dataset1 [dataset2 ...]`.
 
 	a. To run in a `fish` shell it is safer to use the `.fish` alternatives. 

Example: If you want to run `pems` and `cora` on GPU do `./run.sh gpu pems cora`.

*Disclaimer*: Only tested on Linux. However, it should technically run anywhere as long as podman is installed.

# Results
The results will be placed on the `outputs/` folder. 
If you want to skip the training, we make available the complete result used to derive the results in the paper.

Download results: [results.tar.zst](https://figshare.com/s/f1a5ef5e9aa77f5fe18a)

### For the curious!
Inside `run.sh`/`run.fish` you will see the commands used to run the experiments. You can change multiple parameters e.g. training epochs, size of the model, learning rate, etc. by just modifying the commands. The syntax should obey [hydra's](https://hydra.cc/) rules. For a more detailed view of the possibilities you can check `inspector/conf`.


[^1]: If you have any questions, do not hesitate to contact me. Check my website for the correct email to use; I will answer!