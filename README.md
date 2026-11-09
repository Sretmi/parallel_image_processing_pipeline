# High Performance Image Processing Pipeline (HPC)

A high-performance C++ application designed to benchmark and evaluate the performance, scalability, and execution behavior across **Serial (Single-Core Baseline)**, **OpenMP (Shared Memory Multi-Threading)**, and **MPI (Distributed Memory Multi-Processing)** paradigms on large-scale image data.

The pipeline performs a 3-stage image processing sequence using Sobel Edge Detection:
1. **Preprocess:** Color (BGR) to single-channel Grayscale conversion using the weighted formula ($0.299 \times R + 0.587 \times G + 0.114 \times B$).
2. **Process:** 2D spatial convolution computing gradient magnitude ($\sqrt{G_x^2 + G_y^2}$) with $3 \times 3$ horizontal ($G_x$) and vertical ($G_y$) Sobel operator kernels.
3. **Postprocess:** Global binary thresholding to segment edges into a binary (black/white) representation.

---

## 🛠️ Tech Stack & Requirements

- **Language:** C++17
- **Core Dependencies:** OpenCV 4.x (Image I/O, Matrix handling), OpenMP, MPI (MS-MPI / OpenMPI / MPICH)
- **Build System:** CMake ($\ge 3.10$) with GCC, Clang, or MinGW64

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
```
---

## ⚙️ Implementation Details

| Stage | Serial (Baseline) | OpenMP (Shared Memory) | MPI (Distributed Memory) |
| :--- | :--- | :--- | :--- |
| **1. Preprocess (Grayscale)** | Nested loops calculating weighted luminosity per pixel. | `#pragma omp parallel for collapse(2) schedule(static)` — uniform chunk distribution across threads. | Rank 0 loads the image and converts BGR to 1D grayscale buffer. Dimensions are broadcast via `MPI_Bcast`. |
| **2. Process (Sobel 3x3)** | Sequential $3 \times 3$ kernel convolution clamped to $[0, 255]$]. | `#pragma omp parallel for collapse(2) schedule(dynamic)` — dynamic scheduling to mitigate load imbalance during memory access. | Domain decomposition using `MPI_Scatterv`. Each process computes convolution on local row segments. |
| **3. Postprocess (Threshold)** | Sequential binary classification based on threshold value. | `#pragma omp parallel for collapse(2) reduction(+:whitePixelCount)` — race-condition free reduction. | Independent in-place thresholding on local process buffers. |
| **Data Aggregation & Metrics** | Direct file output (`cv::imwrite`). | Direct file output (`cv::imwrite`). | Processed chunks gathered into Rank 0 using `MPI_Gatherv`. `MPI_Reduce(..., MPI_MAX)` measures the slowest process calculation time. |

---

## 🚀 Build & Execution Guide

### 1. Build the Project

Ensure OpenCV and your MPI distribution are configured in your system environment. From the root directory:

```bash
# Create and navigate to the build directory
mkdir build
cd build

# Generate build files and compile all targets
cmake ..
cmake --build .
```

*If building with MinGW on Windows:*
```bash
cmake -G "MinGW Makefiles" ..
mingw32-make
```

---

### 2. Running the Binaries

Execute the binaries directly from the `build/` directory so relative paths resolve to the root input image:

#### Serial Baseline:
```bash
# Linux / macOS:
./serial/serial_app

# Windows:
.\serial\serial_app.exe
```

#### OpenMP (Multi-Threaded):
By default, the application utilizes all available logical processor cores. To set a specific thread count:
```bash
# Linux / macOS:
export OMP_NUM_THREADS=8
./openmp/openmp_app

# Windows (PowerShell):
$env:OMP_NUM_THREADS="8"
.\openmp\openmp_app.exe
```

#### MPI (Multi-Process):
Run using your MPI process manager, specifying the desired process count (`-n`):
```bash
# Linux (OpenMPI / MPICH):
mpirun -n 4 ./mpi/mpi_app

# Windows (MS-MPI):
mpiexec -n 4 .\mpi\mpi_app.exe
```

---

## 📊 Performance Metrics & Output

Each executable benchmarks and logs execution times in milliseconds (`ms`):
- **Preprocess Time:** Wall-clock time taken to read and convert raw color channels to grayscale.
- **Process Time:** Core convolution computation duration. For MPI, this represents the critical path via `MPI_Reduce(MPI_MAX)`.
- **Postprocess Time:** Time taken to apply the binary threshold filter.
- **Communication Overhead (MPI):** The latency and bandwidth cost incurred during scatter (`MPI_Scatterv`), barrier synchronization (`MPI_Barrier`), and gathering operations (`MPI_Gatherv`).
