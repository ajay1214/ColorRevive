# **ColorRevive: AI-Powered Image and Video Colorization** 🎨✨

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange?style=for-the-badge)

## **Overview**
**ColorRevive** explores the application of deep learning to transform grayscale images and videos into realistic, colorized versions. By utilizing pre-trained Convolutional Neural Networks (CNNs) and advanced image-processing techniques, this project demonstrates how AI can bridge the gap between monochrome nostalgia and vibrant modern visuals.

The project features an interactive **Streamlit** web interface, allowing users to upload, process, and download colorized media effortlessly.

---

## **Purpose of the Project**
The goal of ColorRevive is to develop an efficient, user-friendly tool for restoring and enhancing grayscale media. It focuses on applying state-of-the-art deep learning models (specifically the Zhang et al. ECCV 2016 architecture) to automate the colorization process, showcasing the versatility of CNNs in multimedia processing.

---

## **Technologies Used**
* **Python:** Core programming language.
* **OpenCV (cv2):** For image/video processing and DNN (Deep Neural Network) module integration.
* **NumPy:** For matrix operations and numerical computations.
* **Streamlit:** For building the interactive web frontend.
* **Caffe Model Components:**
    * `colorization_release_v2.caffemodel`: Pre-trained weights.
    * `models_colorization_deploy_v2.prototxt`: Network architecture definition.
    * `pts_in_hull.npy`: Cluster centers for the "ab" channels in LAB space.

---

## **How Deep Learning is Used**
The project employs CNNs trained on large datasets (like ImageNet) to predict the **chrominance** ("ab" channels) of a grayscale image based on its **luminance** ("L" channel).

### **The Colorization Process**
1.  **Input:** A grayscale image (L channel).
2.  **Feature Extraction:** The CNN analyzes the image to detect objects, textures, and patterns.
3.  **Prediction:** The model predicts the **'a'** (Green-Red) and **'b'** (Blue-Yellow) channels.
4.  **Reconstruction:** The predicted 'ab' channels are resized and combined with the original 'L' channel.
5.  **Output:** The LAB result is converted to RGB for display.

> **Analogy:** Imagine a black-and-white coloring book. The lines (L channel) tell you *where* objects are. The AI uses its "memory" of the real world to guess which crayons (a and b channels) to use to fill it in!

---

## **Features**

### **Image Colorization**
* **Instant Results:** Upload and view colorized images in seconds.
* **Side-by-Side Comparison:** Compare the grayscale input vs. the AI output.
* **Download:** Save the enhanced image locally.

### **Video Colorization**
* **Frame-by-Frame Processing:** The model iterates through video frames to colorize them individually.
* **Smooth Reconstruction:** Frames are stitched back together to form a cohesive color video.

---

## **Installation and Setup**

### **1. Clone the Repository**
```bash
git clone https://github.com/ajay1214/ColorRevive.git
cd ColorRevive
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Download Model Files**
Create a folder named models in the root directory and download the following files into it:
* **colorization_release_v2.caffemodel** 
👉 Download: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1
* **models_colorization_deploy_v2.prototxt**
* **pts_in_hull.npy**
### **4. Run the Streamlit app:**
```bash
streamlit run app.py
```

Here is the file structure you should have:
```text
ColorRevive/
├── models/
│   ├── colorization_release_v2.caffemodel
│   ├── models_colorization_deploy_v2.prototxt
│   └── pts_in_hull.npy
├── app.py
├── requirements.txt
└── README.md
```

**HAPPY MACHINE LEARNING**

**MADE With ❤️ by Ajay**
