# Garbage Classification with ResNet50

Garbage classification plays an important role in environmental sustainability and smart waste management systems. Automating the sorting of garbage using deep learning techniques can improve recycling efficiency and reduce human labor in waste classification tasks.

This project utilizes **transfer learning** with the **ResNet50** architecture to classify garbage images into multiple categories. The goal is to explore how pre-trained convolutional neural networks can be adapted to solve image classification tasks with limited training data. And compare it with custom CNN.

---

### 🎯 Objective

The main goals of this project are:

- 📈 Evaluating the impact of data augmentation on model generalization.
- ⚙️ Implementing effective preprocessing techniques to improve model performance.
- 📊 Analyzing metrics such as accuracy, F1-score, Precission and Recall to assess results.

---

### 📁 Dataset

- **Type**: Images of garbage categorized into different waste types (glass, cardboard, metal, paper, plastic, trash).
- **Source**: [Kaggle – Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- **Preparation**:
  - Images resized to 224x224 pixels.
  - Applied normalization.
  - Data split into training and validation sets (80/20).
  - Data augmentation (random horizontal flip).

---

### 🧠 Model Architecture

- **Base Model**: Pre-trained ResNet50 from `torchvision.models`.
- **Modifications**:
  - Final fully connected layer replaced with `nn.Linear(2048, 6)` for multi-class classification.
  - Added `Dropout` for regularization.
  - Final layer `nn.Linear(512, 6)`
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam`
- **Metrics**: Accuracy, F1-score, precision, recall

---

### 🛠️ Tools and Libraries

- Python
- PyTorch
- NumPy, Pandas
- OpenCV
- Matplotlib, Seaborn
- Scikit-learn

---
## 🐛 Issue – Shared Dataset Transform Problem

While working on this project, I encountered an interesting and subtle issue related to PyTorch’s `Subset` class.

The problem arises from the fact that `Subset` does **not** create a deep copy of the dataset. Instead, it holds a **reference** to the same underlying dataset object. This means that:

```python
dataset = ImageFolder(root=base_directory)
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)

train_data.dataset.transform = train_transform
val_data.dataset.transform = val_transform
```
At first glance, this looks fine — but here’s the catch:

⚠️ Changing .dataset.transform for val_data also affects train_data, and vice versa, because both are referencing the same underlying dataset!

This caused unintended behavior during training, where both training and validation sets were being transformed in the same way, defeating the purpose of using different preprocessing strategies (e.g., augmentation for train, normalization for val).

✅ Solution
To fix this, I created separate ImageFolder datasets for train and validation directories before applying Subset and transformations:
```python
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_transform)

train_data = Subset(train_dataset, train_indices)
val_data = Subset(val_dataset, val_indices)
```
This ensures that each subset uses its own transformations independently — and avoids unintended side effects.

💡 Lesson Learned
Even small implementation details, like object references, can cause silent bugs in machine learning workflows. Always check how classes like Subset, ImageFolder, or DataLoader manage state — especially with transformations and shuffling!
---

### 📊 Results

| Metric           | ResNet50 | Base Model | Base Model (Aug) | ResNet50 (Aug) |
|------------------|----------|------------|------------------|----------------|
| **Train Accuracy**   | 0.927826 | 0.533611   | 0.296372         | 0.924824       |
| **Train Precision**  | 0.928056 | 0.591371   | 0.326442         | 0.930130       |
| **Train Recall**     | 0.927826 | 0.533611   | 0.296372         | 0.924824       |
| **Train F1**         | 0.927830 | 0.531242   | 0.243155         | 0.927372       |
| **Test Accuracy**    | 0.862123 | 0.566004   | 0.243065         | 0.853331       |
| **Test Precision**   | 0.860099 | 0.542216   | 0.298353         | 0.881586       |
| **Test Recall**      | 0.862123 | 0.566004   | 0.243065         | 0.853331       |
| **Test F1**          | 0.860336 | 0.550020   | 0.181808         | 0.861582       |
---

### 📌 Observations

- ResNet50 performed well out-of-the-box 
- Augmentation helped reduce overfitting and slightly improved generalization.
  
---

### ✅ Conclusion

Transfer learning with ResNet50 provided strong baseline results for garbage image classification. Augmentation and proper preprocessing significantly influenced model performance.

---

### 📦 Future Work

- Experiment with other architectures: **VGG16**, **EfficientNet**, **MobileNetV2**
- Add class weighting or SMOTE because there is quite class  imbalance.
- Deploy model as a web or mobile app for real-time garbage classification.
- Use explainability tools like Grad-CAM to interpret model decisions.

---

### 🙋‍♂️ Author

**Jan Dyndor**  
ML Engineer & Pharmacist  
📧 dyndorjan@gmail.com  
🔗 [Kaggle](https://www.kaggle.com/jandyndor) | [GitHub](https://github.com/Jan-Dyndor) | [LinkedIn](https://www.linkedin.com/in/jan-dyndor/)

---

### 🧠 Keywords

garbage classification, deep learning, computer vision, transfer learning, ResNet50, PyTorch, waste management, image classification
