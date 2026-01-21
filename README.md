![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![CLIP](https://img.shields.io/badge/Model-CLIP-orange.svg)
![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)
![AI](https://img.shields.io/badge/Domain-Multimodal%20AI-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/asif-khan-ak/image_caption_generator/blob/main/image_caption_generator.ipynb
)

# 🖼️ Image Caption Generator with Python

---

## 📌 Project Overview

This project demonstrates a **Multimodal AI system** that can **generate captions for any input image** using **OpenAI’s CLIP (Contrastive Language–Image Pretraining)**.  

Multimodal AI models can understand **multiple types of data simultaneously**, such as text and images. This system compares an image to a set of 254 predefined captions and returns the **top 5 captions that best describe the image** based on **cosine similarity**.  

The project is implemented in **Python** using **PyTorch, Hugging Face Transformers**, and runs seamlessly on **Google Colab**.

---

## 🎯 Objectives

- Load and preprocess images for CLIP input  
- Extract **semantic embeddings** of images using CLIP  
- Compare image embeddings with multiple text embeddings (captions)  
- Calculate **cosine similarity** to find the best matching captions  
- Demonstrate real-world applications like:
  - Automated image captioning for social media  
  - Visual search engines  
  - AI-powered content recommendation

---

## 📂 Dataset / Captions

The project uses a **custom list of 254 captions** to match against input images.  

**Example captions:**  
- "The horizon holds endless possibilities."  
- "Where tradition meets tranquility."  
- "Your moment of serenity."  
- "The tide carries secrets ashore."  
- "A haven for your senses."

> ⚠ Note: No external datasets are used. You can expand the captions list to suit different applications.

---

## 🔄 Project Pipeline

1. **Image Loading and Preprocessing**  
   - Load the input image using **Pillow**  
   - Convert it to **RGB** format  
   - Tokenize and preprocess with **CLIPProcessor**  

2. **Extract Image Embeddings**  
   - Use **CLIPModel** to get a semantic embedding of the image  
   - Represents the image in a 512-dimensional **feature space**

3. **Text Embeddings for Captions**  
   - Convert all captions to embeddings using the same CLIP model  
   - Ensures both image and text are in the **same semantic space**

4. **Calculate Cosine Similarity**  
   - Measure similarity between image features and each caption  
   - Rank captions by similarity to select the **top 5 matches**

5. **Return Best Matching Captions**  
   - Display captions with similarity scores

---

## 🧠 Model Details

### 🔹 CLIP Model
- **Pretrained model:** `openai/clip-vit-base-patch32`  
- **Architecture:** Vision Transformer + Text Transformer  
- **Embedding Dimension:** 512  
- **Framework:** PyTorch  
- **Tokenizer:** Hugging Face CLIPProcessor  

> CLIP allows **direct comparison of images and text** in the same embedding space.

---

**Output (for an image of the ocean):**

1. The horizon holds endless possibilities. (Similarity: 0.2496)
2. Where tradition meets tranquility. (Similarity: 0.2448)
3. Your moment of serenity. (Similarity: 0.2423)
4. The tide carries secrets ashore. (Similarity: 0.2422)
5. A haven for your senses. (Similarity: 0.2363)

---

## 🖥️ Tech Stack

* Python | PyTorch | Hugging Face Transformers
* Pillow | NumPy | Pandas | Matplotlib
* Google Colab for interactive notebooks

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/image_caption_generator.git
cd image_caption_generator
```

Open the notebook `image_caption_generator.ipynb` in **Google Colab** and run all cells.

---

## 👤 Author

**Asif Khan**
Data Science & AI Enthusiast

---

## ⭐ Acknowledgments

* Inspired by OpenAI’s **CLIP model**
* Hugging Face Transformers & PyTorch community for **seamless multimodal AI tools**
