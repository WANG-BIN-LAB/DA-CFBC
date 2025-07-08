# 🦷 DA-CFBC: Direction-Aware and Center-Focused Boundary-Constrained Framework for CBCT 

This repository provides the official implementation of our paper:

> **DA-CFBC: Direction-Aware and Center-Focused Boundary-Constrained Framework for CBCT**  
> _Bin Wang, et al._  


We propose a novel 3D deep learning framework for **tooth instance segmentation** in Cone Beam CT (CBCT) images. Our model introduces a direction-aware convolution module and a center-focused boundary constraint strategy to effectively address challenges like directional variation, metal artifacts, and inter-tooth adhesion.

---

## 📌 Hightlights

- ✅ Direction-aware modules to capture anatomical orientation
- ✅ Center-focused Boundary-Constrainted to separate adjacent teeth
- ✅ Boundary Optimization algorithm to refine segmentation results


---

## 🚀 Training

python train_identification.py 
python train_segmentation.py 

## 🚀 Inference & Evaluation
python lets_seg.py

#  📊 Results
Method	DSC ↑ Iou↑	HD ↓	ASD ↓
DA-CFBC (Ours)	0.977	0.954 1.013	0.233

For more detailed results and ablation studies, please refer to the paper.
