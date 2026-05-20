
## Using Results and Reproducing Figures

After training the models, results will be stored in the `outputs/` directory. Each experiment contains a `stats/` folder with all computed metrics.

### Visualizing Results
- You can use the plotting scripts in the `figures/` directory.
- Organise the content of the `outputs/stats/` directory to match the one present in the plotting scripts (apologies, the plotting scripts were not done with automation in mind).

### Reproducing Paper Figures (Recommended)
- For convenience, we provide pre-organized metrics that match the required directory structure for figure generation.
- Using these avoids manual configuration and ensures consistency.

### Verification
- You can compare the provided metrics with your own training outputs to confirm they match.

---

Download organised results: [do_figures.tar.gz](https://figshare.com/s/f1a5ef5e9aa77f5fe18a)

1. Extract and place the `do_figures` directory in the same parent directory of `gnn-uq-inspector`.
2. Run the commands

```
python make_plots_raw_metrics.py --type nll --misc_type soup --datasets artnetviews chameleon gapsmallqm9

python make_plots_raw_metrics.py --type point --misc_type soup --datasets artnetviews chameleon gapsmallqm9

python make_plots_raw_metrics.py --type point --misc_type soup --datasets tolokers2 cora citeseer

python make_plots_raw_metrics.py --type nll --misc_type soup --datasets tolokers2 cora citeseer

python make_plots_raw_metrics.py --type nll --misc_type normal --datasets chameleon artnetviews gapsmallqm9

python make_plots_raw_metrics.py --type nll --misc_type normal --datasets tolokers2 cora citeseer

python make_plots_raw_metrics.py --type nll --misc_type normal --datasets chameleon artnetviews gapsmallqm9

python make_plots_raw_metrics.py --type point --misc_type ood  --datasets artnetviews tolokers2

python make_plots.py --datasets cora citeseer tolokers2 --type nll

python make_plots.py --datasets cora citeseer tolokers2 --type point

python make_plots.py --datasets artnetviews chameleon gapsmallqm9 --type nll

python make_plots.py --datasets artnetviews chameleon gapsmallqm9 --type point

python make_plots_new_nll.py --datasets cora citeseer tolokers2

python make_plots_new_nll.py --datasets chameleon artnetviews gapsmallqm9
```

3. Switch OOD to True in `make_plots_new_nll.py` and `python make_plots_new_nll.py --datasets tolokers2 artnetviews`

4. To do Figure 1, the PEMS map, we used the publicaly available code at [V. Borovitskiy. PeMS Regression: A Benchmark Suite for Node Regression with Uncertainty](https://github.com/vabor112/pems-regression/tree/main/pems_regression/).
