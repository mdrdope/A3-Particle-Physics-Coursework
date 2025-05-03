# Project Overview  
This project addresses coursework tasks for the MPhil in Data Intensive Science (Lent Term 2025) at the University of Cambridge. It includes three sections:

✅ Section A (A2): Implementation of rotation-equivariant convolutional layers using custom Keras layers, tested on rotated MNIST images.

✅ Section B (B1): Analysis of ATLAS Open Data for detecting Higgs boson signals via statistical fitting techniques.

✅ Section C (C2): Development of a Transformer-based neural network for classifying jet flavors from simulated Z⁰ boson decay events.

Each section includes clearly structured Jupyter notebooks, organized datasets, trained models, and reproducible experimental pipelines.

# Folder and File Structure
Below is the tree structure of the project:

## Whole structure
```plaintext
ym432/
├── A3_SectionA_Q2/
├── A3_SectionB_Q1/
├── A3_SectionC_Q2/
├── report/
│   └── A3_Coursework_Report_ym432.pdf
├── .gitignore
├── LICENSE
├── readme.md
└── requirements.txt
```

Description:
- `A3_SectionA_Q2/`: Implementation of rotation-equivariant convolutional layers with associated Keras models and evaluation notebooks.
- `A3_SectionB_Q1/`: Data analysis scripts and datasets used for statistical fitting to detect Higgs boson signals in ATLAS Open Data.
- `A3_SectionC_Q2/`: Event classification neural network, including preprocessing, training scripts, datasets, and trained Transformer models.
- `report/`: Coursework report in PDF format.



## Section A Structure

```plaintext
A3_SectionA_Q2/
├── models/
│   ├── mlp_exp1.weights.h5
│   ├── mlp_exp2.weights.h5
│   ├── rot_exp1.weights.h5
│   └── rot_exp2.weights.h5
└── SectionA_Q2.ipynb
```

Description:
- `models/`: Trained model weights for baseline multilayer perceptron (MLP) and rotation-equivariant convolutional neural network (CNN).
- `SectionA_Q2.ipynb`: Jupyter notebook providing the full experimental pipeline for Section A2.

## Section B Structure
```plaintext
A3_SectionB_Q1/
├── dataset/
│   ├── data15_periodD.root
│   ├── ...
│   ├── data16_periodL.root
│   ├── mc_341081.ttH125_gamgam.GamGam.root
│   ├── ...
│   └── mc_345319.ZH125J_Zincl_gamgam.GamGam.root
└── SectionB_Q1.ipynb
```

Description:
- `dataset/`: ATLAS Open Data ROOT files, including both experimental collision data (`data*.root`) and simulated Monte Carlo Higgs events (`mc*.root`).
- `SectionB_Q1.ipynb`: Jupyter notebook pipeline. To run, manually place 21 `data*.root` and 5 `mc_*.root` files into `dataset/`.


## Section C Structure

```plaintext
A3_SectionC_Q2/
├── dataset/
│   ├── info.md
│   ├── Zbb.root
│   ├── Zcc.root
│   ├── zqq_flavor_v2.h5
│   ├── zqq_flavor.h5
│   └── Zss.root
├── results/
│   ├── clean_hist/
│   ├── raw_hist/
│   ├── model/
│   │   ├── best_model_v2.keras
│   │   └── best_model.keras
│   ├── training_history.csv
│   ├── training_history_v2.csv
│   ├── variable_clean_stats.json
│   └── variable_raw_stats.json
├── src/
│   ├── model.py
│   └── dataloader.py
├── SectionC_Q2_1data_preprocessing.ipynb
└── SectionC_Q2_2training.ipynb
```

Description:
- `dataset/`: Preprocessed datasets (`zqq_flavor.h5`, `zqq_flavor_v2.h5`) and info (`info.md`).  Please manually place the required `.root` files in this folder, then run `SectionC_Q2_1data_preprocessing.ipynb` to generate the `.h5` files.
- `results/`: Trained models, training logs, preprocessing statistics, and histograms.
    - `model/`: Transformer model checkpoints (`best_model.keras`, `best_model_v2.keras`).
    - clean_hist/: Preprocessing histograms.
    - raw_hist/: Preprocessing histograms.
    - `training_history.csv`, `training_history_v2.csv`: Logs of training metrics.
    - `variable_clean_stats.json`, `variable_raw_stats.json`: Feature statistics from preprocessing.
- `src/`: Python modules for data loading and model definition.
- `SectionC_Q2_1data_preprocessing.ipynb`: Notebook for generating `.h5` datasets. To run, place three `.root` files manually in `dataset/`.
- `SectionC_Q2_2training.ipynb`: Notebook for model training using preprocessed `.h5` data.



# Environment Setup

1. **Using Conda**:

   - Navigate to the directory containing `requirements.txt`:
     ```bash
     cd <path to requirements.txt>
     ```

   - Create a virtual environment:
     ```bash
     conda create -n <venv_name> python=3.10 -y
     ```

   - Activate the virtual environment:
     ```bash
     conda activate <venv_name>
     ```

   - Install the dependencies:
     ```bash
     pip install --no-cache-dir -r requirements.txt
     ```

2. **Select the Environment in Your Notebook**:
   - In your Jupyter Notebook (e.g., via JupyterLab or VS Code's Notebook interface), ensure that you select `<venv_name>` as the active kernel to run your code.


# Notes
- To fully reproduce results for **Section B** and **Section C**, manually place the required `.root` data files into the respective `dataset/` directories before running the notebooks.

# Declaration of Auto Generation Tools

This project leverages AI tools to assist in the development process. Specifically:
- **Code**: Portions of the code were generated using ChatGPT-4o based on pseudocode and instructions provided by the author.
- **Report**: The project report and documentation were created using ChatGPT-4o, guided by the author's detailed instructions.

However, all ideas, concepts, and the overall project structure are entirely the author's own.


# License
This project is licensed under the terms specified in the `LICENSE` file.