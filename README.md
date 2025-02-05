# ALOG Framework Execution

This repository contains the necessary scripts to run the **ALOG framework** for geospatial data analysis using **Local Differential Privacy (LDP)**. The framework processes four datasets (**uniform, normal, geo, and porto**) and saves the results in a `.csv` file.

## Execution

The **`run.sh`** script automates the execution of the ALOG framework by running four different test scripts, each with distinct parameters. These scripts perform frequency estimation and grid adaptation under LDP, ensuring privacy preservation while maximizing data utility.

## Steps to Run

1. **Grant execution permissions** (if needed):
   ```bash
   chmod +x run.sh
   ```

2. **Run the framework:**
   ```bash
   ./run.sh
   ```

## Process Overview

- **Dataset Processing:** The framework will sequentially process four datasets:
  - `uniform`: Simulates evenly distributed user movements.
  - `normal`: Simulates user movement following a normal distribution.
  - `geo`: Real-world geospatial dataset with diverse movement patterns.
  - `porto`: Taxi trajectory dataset from Porto.

- **Execution of Test Scripts:**
  - The script runs **four different test scripts**, each with different configurations of privacy budgets, grid sizes, and adaptation parameters.
  - Each test script applies **Local Differential Privacy (LDP)** mechanisms and stores the processed results.

- **Results Storage:**
  - The processed data and evaluation metrics are saved in a **`.csv` file** for further analysis.

## Notes
- Ensure sufficient computational resources, as processing may require extended runtime depending on the dataset size.
- The **Grid Adaptation Window (GAW)** mechanism optimizes the grid refinement process for improved data utility.

For any issues, please check the logs or contact the maintainers.


