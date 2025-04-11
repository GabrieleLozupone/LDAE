### Derive from the base container image
### We use tensorflow base image because the preprocessing pipeline
### contains antspynet tf-based brain extraction model
FROM tensorflow/tensorflow:latest-gpu

# Install required dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your BrainDiffAE project into the image
COPY . /app/LDAE

# Set the working directory to your project
WORKDIR /app/LDAE

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install JupyterLab
RUN pip3 install jupyterlab

# Expose a volume for your datasets
VOLUME ["/data"]

# Expose a volume for the project
VOLUME ["/app/LDAE"]

# Expose Jupyter port
EXPOSE 8888

# Default command to keep the container running
CMD ["bash"]