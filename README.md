# Deep Learning Pipeline for Gene Expression Analysis

## Overview
This project provides a **deep learning pipeline** to analyze **gene expression data** in cancer using **TensorFlow/Keras and PyTorch**. The pipeline includes:
- **Data Preprocessing**: Standardizing gene expression values.
- **Deep Learning Models**: Implemented in both **TensorFlow/Keras** and **PyTorch**.
- **Training and Evaluation**: Tracks model performance.
- **Visualization**: Plots accuracy and loss curves.
- **Model Saving**: Stores trained models for future use.

## Project Structure
```
Cancer_Gene_Expression_Analysis/
│── data/                   # Raw and processed datasets
│── figures/                # Saved plots and visualizations
│── notebooks/              # Jupyter notebooks for analysis
│── src/                    # Python scripts for pipeline
│── results/                # Outputs and final reports
│── README.md               # Project documentation
│── requirements.txt        # Dependencies
│── main.py                 # Main execution script
│── config.yaml             # Configuration file
│── LICENSE                 # License information
```

## Installation
### Prerequisites
Ensure you have Python (>=3.8) installed. Then, install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
Modify `config.yaml` to customize parameters such as:
```yaml
train:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  model_type: "keras"  # Options: keras, pytorch
```

## Running the Pipeline
To train and evaluate the models, run:
```bash
python main.py
```

## Expected Outputs
- **Trained model saved in `results/`**
- **Accuracy and loss plots in `figures/`**
- **Performance metrics printed in console**

## Contributing
Feel free to submit a **pull request** or open an **issue**.

## License
This project is licensed under the MIT License.

## Contact
**Etienne Ntumba Kabongo**
Email: [etienne.ntumba.kabongo@umontreal.ca](mailto:etienne.ntumba.kabongo@umontreal.ca)

---

# requirements.txt
```
numpy
pandas
matplotlib
seaborn
tensorflow
keras
torch
torchvision
torchaudio
scikit-learn
pyyaml
```

---

# config.yaml
```yaml
data:
  dataset_path: "data/gene_expression.csv"
train:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  model_type: "keras"  # Options: keras, pytorch
  save_model_path: "results/"
```
