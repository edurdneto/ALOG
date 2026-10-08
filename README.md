# ALOQ: Adaptive Longitudinal Quadtree

Code for the experiments of **"Longitudinal Geospatial Frequency Estimation under Adaptive Local Differentially Private Model"** (ALOQ). ALOQ estimates how often users visit each region of a map over many timestamps under Local Differential Privacy (LDP). It uses a quadtree that adapts to where users are, using only privatized reports.


## Requirements

Python 3.10 or newer. Install the dependencies with:

```bash
pip install -r requirements.txt
```



## Datasets

The script expects these files:

| Dataset | Code | File |
|---|---|---|
| S1, synthetic uniform | `u` | `Dataset/Sintetic_Uniform/data_uni.pkl` |
| S2, synthetic normal | `n` | `Dataset/Sintetic_Normal/data_norm.pkl` |
| GeoLife | `g` | `Dataset/Geolife_Trajectories_Dataset/Taxi/geolife_cartesian_bounded_120.pkl` |
| Porto taxi | `p` | `Dataset/Taxi_Porto_KAGGLE/new_taxi_portugal_10000_120.pkl` |

Each file is a pickled list with one entry per user. Each entry is that user's list of `(x, y)` positions in Cartesian coordinates, one per timestamp. The synthetic files are in the repository. The GeoLife and Porto files must be placed at the paths above.

## Running

```bash
chmod +x run.sh
./run.sh                 # the four datasets
./run.sh uniform porto   # only some of them (uniform, normal, geo, porto)
```

Or run one dataset directly:

```bash
python3 aloq.py profiles/geo.txt
```

The tasks run in parallel on all CPU cores but one. If a run is interrupted, run the same command again: tasks that already have a CSV are skipped.

## Results

```
Results/
├── Experiment_Uniform/
│   ├── Budget/        # privacy budget vs. utility
│   ├── Granularity/   # initial quadtree size (k) vs. utility
│   └── Numpoints/     # reports per user vs. utility
├── Experiment_Normal/
├── Experiment_Geo/
└── Experiment_Porto/
```

Each folder has one CSV per run and a `profile.csv` with the parameters used. The main CSV columns are:

| Column | Meaning |
|---|---|
| `method` | LDP protocol: `LOSUE`, `LOLOHA` or `RAPPOR` |
| `structure` | Approach (see below) |
| `budget` | Privacy budget ε per report |
| `grid_size_base` | Leaves of the initial quadtree (k) |
| `grid_size` | Leaves at each timestamp |
| `mse_avg`, `mae_avg` | Average MSE and MAE over the timestamps |
| `mse_t`, `mae_t` | MSE and MAE at each timestamp |
| `final_budget` | Average total privacy budget consumed per user |
| `hits` | Average memoization hits |
| `num_points` | Reports per user (timestamps) |

### Approaches

| `structure` | Name in the paper |
|---|---|
| `AP2` | ALOQ-2S (two-stage sanitization) |
| `AP3` | ALOQ-1S-Adap (one stage, previous quadtree) |
| `AP4` | ALOQ-1S-Base (one stage, base quadtree) |
| `Uniform` | Static uniform quadtree, used by the RAPPOR, L-OSUE and LOLOHA baselines |
| `PRIVAG` | PrivTC (adaptive grid PrivAG), adapted to the longitudinal setting |

`LOLOHA` and `RAPPOR` run only with `Uniform`. `PRIVAG` runs only with `LOSUE`.

## Profiles

The experiments are set up in `profiles/<dataset>.txt`. Each line holds `key:value` pairs separated by `;`, and lists of values are separated by `,`. Lines starting with `#` are comments.

The **first line** describes the dataset:

```
s:1;d:g;
```

| Key | Meaning |
|---|---|
| `s` | Number of seeds (repetitions) |
| `d` | Dataset: `u`, `n`, `g` or `p` |

Every **following line** is one experiment, saved in its own folder:

```
folder:Budget;k:256;e:0.05,0.1,0.3,0.6;e_prop:0.3;w:4;p:40;structures:Uniform,AP2,AP3,AP4,PRIVAG;methods:LOSUE,LOLOHA,RAPPOR
```

| Key | Meaning |
|---|---|
| `folder` | Sub-folder of the results |
| `k` | Leaves of the initial uniform quadtree. Must be a power of 4 (16, 64, 256, 1024…). The cell size is computed from the data bounds. |
| `g` | Alternative to `k`: cell side length, in the data units |
| `e` | Privacy budgets ε |
| `e_prop` | Share of the budget used in the first stage of ALOQ-2S |
| `w` | Window parameter (recorded in the results) |
| `p` | Reports per user (timestamps) |
| `structures` | Approaches to run (default `AP2`) |
| `methods` | LDP protocols (default `LOSUE`) |

The profiles in `profiles/` reproduce the experiments of the paper. The fixed ALOQ parameters are set in `aloq.py`: similarity threshold 0.7, similarity window 3, α = 0.1, and budget reduction factor 0.1 (0.6 for LOLOHA).

