
# Chest X-ray Disease Classification with DenseNet121

This project trains a deep learning model using transfer learning (DenseNet121) to classify chest X-ray images into 15 categories (14 disease states + "No_Finding") using the NIH Chest X-ray dataset.

---

## 📁 Dataset

- **Source**: NIH Chest X-ray dataset (provided on HiPerGator)
- **Path**: `/lustre/fs0/bsc4892/share/ChestXray-NIHCC/images_14_cat`
- **Classes**: 15 total (including "No_Finding")

---

## 🧠 Model & Approach

- **Architecture**: DenseNet121 with pretrained ImageNet weights
- **Modification**: Final classifier layer adjusted to output 15 classes
- **Loss Function**: CrossEntropyLoss with class balancing via `WeightedRandomSampler`
- **Optimizer**: Adam (lr=1e-4)

---

## 🔁 Training Details

- **Image Size**: 224 x 224
- **Epochs**: 5
- **Batch Size**: 32
- **Transforms**:
  - Resize
  - ToTensor
  - Normalize (ImageNet stats)

---

## 📊 Evaluation

- **Confusion Matrix**: Plotted after model evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score via `classification_report`

---

## 🧪 Reproducibility

### Dependencies
See `requirements.txt` for all required libraries. Key ones include:
- `torch`
- `torchvision`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tqdm`

---

## 📝 Notes

- The model uses a weighted sampler to handle class imbalance.
- ROC curves can be added by calculating per-class AUC using `roc_auc_score` from `sklearn`.

---

## 📂 Files in this Repository

- `chest_xray_densenet.ipynb` – Full training and evaluation notebook
- `README.md` – This file
