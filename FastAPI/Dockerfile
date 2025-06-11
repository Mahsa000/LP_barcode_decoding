# Use a lightweight Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for compiling Cython extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    qt5-qmake \
    qtbase5-dev \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*  # Cleanup to reduce image size

# Copy the requirements file first and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --only-binary=h5py h5py

# Install NumPy and Cython before running setup.py
RUN pip install --no-cache-dir cython numpy

# Ensure NumPy is available when building Cython extension
RUN python -c "import numpy; print(numpy.get_include())"

# Copy the entire project
COPY . .

# Build and install the Cython extension
WORKDIR /app/lase_analysis/c_code/utils
RUN python setup.py build_ext --inplace
RUN pip install .

# Return to app directory
WORKDIR /app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app3_march5:app", "--host", "0.0.0.0", "--port", "8000"]
