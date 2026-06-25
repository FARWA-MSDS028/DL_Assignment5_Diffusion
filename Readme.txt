================================================================
README.txt
Deep Learning Assignment 5 (Bonus) - Diffusion Model
Name: Farwa | Roll Number: MSDS25028
================================================================

PROJECT DESCRIPTION:
This project implements a DDPM (Denoising Diffusion Probabilistic Model)
for generating animal images using PyTorch.

================================================================
FOLDER STRUCTURE:
================================================================
Farwa_MSDS25028_05/
├── MSDS25028_05.py          -- Main training code (full version with comments)
├── MSDS25028_05_allCode.py  -- ALL code combined in one file (required)
├── test_single_sample.ipynb -- Jupyter notebook for evaluation/demo
├── Readme.txt               -- This file
├── Report.pdf               -- Assignment report
├── saved_models/
│   └── diffusion_model.pth  -- Saved trained model weights
└── results/
    ├── forward_process.png  -- Shows noise being added at different timesteps
    ├── loss_curve.png       -- Training loss graph
    ├── final_generated.png  -- Final generated animal images
    └── epoch_*_samples.png  -- Samples generated during training

================================================================
REQUIREMENTS - INSTALL THESE FIRST:
================================================================
Open VS Code terminal (Ctrl + `) and run:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install matplotlib pillow numpy

(If you have a GPU, install the GPU version of PyTorch from pytorch.org)

================================================================
HOW TO RUN:
================================================================

STEP 1 - Prepare your dataset:
    - Unzip animal_datasets.zip
    - Place the folder at: ./animal_datasets/
    - Structure should be:
        animal_datasets/
            cat/
                img1.jpg
                img2.jpg
                ...
            dog/
                img1.jpg
                ...
            (other classes...)

STEP 2 - Train the model:
    python MSDS25028_05.py --dataset ./animal_datasets

    With custom settings:
    python MSDS25028_05.py --dataset ./animal_datasets --epochs 100 --batch_size 4 --lr 0.0002

    Arguments:
    --dataset       Path to dataset folder (REQUIRED)
    --epochs        Number of training epochs (default: 50)
    --batch_size    Batch size (default: 8, use 4 if memory issues)
    --lr            Learning rate (default: 0.0002)

STEP 3 - Generate images only (after training):
    python MSDS25028_05.py --dataset ./animal_datasets --generate_only

STEP 4 - For viva evaluation, open Jupyter notebook:
    jupyter notebook test_single_sample.ipynb
    (Run cells one by one)

================================================================
EXPECTED OUTPUT:
================================================================
- results/forward_process.png  : Shows original -> noisy images at t=0,100,250,500,750,999
- results/loss_curve.png       : Training loss decreasing over epochs
- results/final_generated.png  : 8 generated animal-like images
- saved_models/diffusion_model.pth : Saved model weights

================================================================
COMMON ERRORS & FIXES:
================================================================

Error: "No module named torch"
Fix: pip install torch torchvision

Error: "No class folders found in dataset"
Fix: Check your dataset path. Run:
     python MSDS25028_05.py --dataset C:/full/path/to/animal_datasets

Error: "CUDA out of memory"
Fix: Use --batch_size 2 or --batch_size 4

Error: "drop_last error" with small dataset
Fix: Normal - the last incomplete batch is dropped

Error: Jupyter notebook not found
Fix: pip install jupyter, then: jupyter notebook

================================================================
MODEL ARCHITECTURE:
================================================================
- Model: U-Net with Skip Connections
- Timestep Embedding: Sinusoidal Position Embedding
- Activation: SiLU (Sigmoid Linear Unit)  
- Normalization: GroupNorm (8 groups)
- Loss: Custom L2 (MSE) between actual and predicted noise
- Diffusion Steps: T=1000
- Noise Schedule: Linear beta schedule (1e-4 to 0.02)
- Optimizer: Adam with Cosine Annealing LR

================================================================