
# 🫁 Pneumonia Detection from Chest X-rays

A deep learning-based tool for detecting pneumonia in chest X-ray images using a fine-tuned ResNet50 model. The goal? Support faster, more accessible diagnostic assistance, especially in resource-limited healthcare settings.

## 🔍 Problem Statement

Pneumonia is a serious lung infection that can be life-threatening if not diagnosed early. Radiologists often use chest X-rays to detect it, but reading and interpreting these images can be time-consuming and subjective.

This project uses convolutional neural networks (CNNs) to automatically classify X-ray images as either:

- **Normal**
- **Pneumonia**

## 🧠 Approach

I trained a CNN using **transfer learning** with **ResNet50**, a powerful deep learning model pre-trained on ImageNet. Here's what the pipeline looks like:

- ✅ Data preprocessing + augmentation (rotation, flipping, zooming)
- ✅ Normalization and resizing (224x224 RGB)
- ✅ Transfer learning with frozen base layers
- ✅ Fine-tuning top layers
- ✅ Validation with a held-out test set

### Model Highlights

- 📦 Base Model: `ResNet50` (pretrained on ImageNet)
- 🧪 Training: On Kaggle’s Chest X-ray dataset
- 🎯 Evaluation Metric: Validation accuracy + loss

## 📊 Results

| Metric              | Value        |
|---------------------|--------------|
| Validation Accuracy | **92.13%**   |
| Validation Loss     | **0.21**     |

The model performs strongly and generalizes well on unseen chest X-ray images.

## 🔗 Useful Links

- 📁 **Dataset**: [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 📓 **Kaggle Notebook**: [View Notebook](https://www.kaggle.com/code/mohammad007kh/chest-x-ray-pneumonia-classifier-resnet50/)
- 💻 **GitHub Repository**: [View Repo](https://github.com/mohammad007kh/Chest-X-ray-Pneumonia-Classifier-ResNet50-)

## 🖥️ Try it Yourself (Streamlit App – Coming Soon)

> 🧪 You'll be able to upload a chest X-ray image and get an instant prediction once the app is live. Stay tuned!

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/mohammad007kh/Chest-X-ray-Pneumonia-Classifier-ResNet50-.git
cd Chest-X-ray-Pneumonia-Classifier-ResNet50-
```

### 2. Run the Streamlit App (Coming Soon)

```bash
streamlit run app/app.py
```

### 3. Upload an Image

Use a chest X-ray in JPG/PNG format and the model will predict whether it indicates **Pneumonia** or **Normal**.

## 🙋‍♂️ Why This Project?

This was built as part of my portfolio to demonstrate skills in:

- Data preprocessing & augmentation
- Transfer learning
- Model evaluation
- Deployment using Streamlit

Feel free to use this as a reference, and I’d love feedback or collaboration ideas!

## 🤝 Contributions & Feedback

Open to suggestions, pull requests, or collaboration! Let's build better tools for better healthcare.

---

*Built with 💙 by [Mohammad Khoddami](https://github.com/mohammad007kh)*
