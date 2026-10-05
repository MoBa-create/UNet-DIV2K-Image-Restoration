# 🖼️ HD Image Reconstruction with U-Net on DIV2K

A PyTorch-based Deep Learning project implementing a 256x256 **U-Net Autoencoder** architecture trained on the high-resolution **DIV2K dataset**. This model learns to reconstruct high-resolution images while retaining crisp details, edges, and fine features without severe compression loss.

---

## 🌟 Key Features

* **High-Resolution Processing:** Trains on $256 \times 256$ HD patches (Random Crop) instead of low-res datasets like CIFAR-10 ($32 \times 32$).
* **Automated Dataset Setup:** Automatically downloads and extracts the DIV2K dataset on the first run.
* **Optimized Hardware Execution:** Configured with multi-processing workers and CUDA memory pinning for modern GPUs.
* **Full Inference Pipeline:** Includes test script for evaluating model performance on custom test images.

---

<img width="1245" height="646" alt="Screenshot 2026-10-05 030732" src="https://github.com/user-attachments/assets/bca6023e-d71a-4bc8-b704-4884e3990db8" />


## 🏗️ Architecture

The model uses a symmetric **U-Net** design:
* **Encoder:** 3 Convolutional stages with BatchNorm and ReLU, reducing spatial dimensions from $256 \times 256$ to $32 \times 32$ bottleneck.
* **Bottleneck:** 512-channel feature extractor.
* **Decoder:** 3 Transposed Convolutional stages with **Skip Connections** concatenated from corresponding encoder stages.
* **Output Layer:** Sigmoid activation producing a $256 \times 256 \times 3$ RGB output tensor.

---

## 📁 Repository Structure

```text
├── train_unet_div2k.py   # Training script with DIV2K dataset pipeline & UNet256
├── test_hd_image.py      # Inference script to visualize output on custom images
├── requirements.txt      # Project dependencies
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
