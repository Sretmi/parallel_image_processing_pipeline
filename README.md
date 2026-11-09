# High Performance Image Processing Pipeline (HPC)

A high-performance C++ application designed to benchmark and evaluate the performance, scalability, and execution behavior across **Serial (Single-Core Baseline)**, **OpenMP (Shared Memory Multi-Threading)**, and **MPI (Distributed Memory Multi-Processing)** paradigms on large-scale image data.

The pipeline performs a 3-stage image processing sequence using Sobel Edge Detection:
1. **Preprocess:** Color (BGR) to single-channel Grayscale conversion using the weighted formula ($0.299 \times R + 0.587 \times G + 0.114 \times B$)[cite: 4, 6].
2. **Process:** 2D spatial convolution computing gradient magnitude ($\sqrt{G_x^2 + G_y^2}$) with $3 \times 3$ horizontal ($G_x$) and vertical ($G_y$) Sobel operator kernels[cite: 4, 6, 8].
3. **Postprocess:** Global binary thresholding to segment edges into a binary (black/white) representation[cite: 4, 6, 8].

---

## 🛠️ Tech Stack & Requirements

- **Language:** C++17[cite: 1, 3]
- **Core Dependencies:** OpenCV 4.x (Image I/O, Matrix handling), OpenMP, MPI (MS-MPI / OpenMPI / MPICH)[cite: 1, 3]
- **Build System:** CMake ($\ge 3.10$) with GCC, Clang, or MinGW64[cite: 1, 3]

---

## 📂 Project Architecture

```text
parallel_image_processing_pipeline/
├── mpi/
│   ├── CMakeLists.txt     # Target configuration for MPI executable
│   └── main.cpp           # Domain Decomposition via 1D row slicing (Scatterv / Gatherv)
├── openmp/
│   ├── CMakeLists.txt     # Target configuration with OpenMP flags
│   └── main.cpp           # Shared memory multi-threading (collapse, scheduling, reduction)
├── serial/
│   ├── CMakeLists.txt     # Target configuration for serial baseline
│   └── main.cpp           # Single-threaded reference implementation
├── CMakeLists.txt         # Root build configuration
├── input.jpg              # High-resolution benchmark input image
└── README.md
