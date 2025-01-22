# 🧠 Alzheimer's MRI Classification with Augmented Dataset

This project leverages Convolutional Neural Networks (CNNs) to classify MRI scans into four categories based on the **Augmented Alzheimer's MRI Dataset**. The workflow includes preprocessing, data augmentation, model training, and evaluation to achieve accurate predictions.

---

## 📂 Dataset
- **Source**: [Augmented Alzheimer's MRI Dataset on Kaggle]
- **Classes**:
  - **Mild Demented**
  - **Moderate Demented**
  - **Non-Demented**
  - **Very Mild Demented**
- **Structure**:
  - Training, validation, and test splits created dynamically:
    - **80%** for Training
    - **10%** for Validation
    - **10%** for Testing

---

## 🛠 Tools and Frameworks
- **Programming Language**: Python
- **Libraries**:
  - TensorFlow & Keras
  - NumPy & Pandas
  - Matplotlib & Seaborn
  - Scikit-learn
- **Platform**: Kaggle Hub

---

## 🔍 Project Workflow
1. **Dataset Loading**:
   - Downloaded and structured the `OriginalDataset` directory.
   - Utilized TensorFlow's `image_dataset_from_directory` for efficient image loading.
   
2. **Data Preprocessing**:
   - Resized images to **200 x 200 pixels**.
   - Normalized pixel values to **[0,1]**.

3. **Data Augmentation**:
   - Applied transformations to improve robustness:
     - Random rotation
     - Zoom
     - Contrast adjustments
     - Horizontal and vertical flips

4. **Model Architecture**:
   - **Sequential CNN** with the following:
     - **6 Convolutional layers**: Filters range from 32 to 128
     - **MaxPooling layers**: Reducing spatial dimensions
     - **Dropout layers**: Preventing overfitting
     - **Dense layers**: For feature extraction and classification
   - **Output**: Softmax activation for multi-class classification

5. **Training**:
   - **Optimizer**: Adam
   - **Loss Function**: Categorical Crossentropy
   - **Metrics**: Accuracy
   - **Epochs**: 85 (Changing #)

6. **Evaluation**:
   - Analyzed model performance using:
     - Accuracy
     - Confusion Matrix
   - Visualized training vs validation accuracy and loss.

---

## 📊 Results
- **Test Accuracy**: Achieved Train Accuracy of **82%**
- **Confusion Matrix**: Highlighted class-specific performance.

---

## 🖼 Visualizations
1. **Data Exploration**:
   - Displayed sample images with class labels.
2. **Training Performance**:
   - Plotted accuracy vs epochs and loss vs epochs.
3. **Confusion Matrix**:
   - Illustrated misclassifications and strengths.
