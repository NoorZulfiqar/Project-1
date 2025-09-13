# Use Python 3.11 (works with dlib + face-recognition)
pip install --upgrade pip
FROM python:3.11

# Install system dependencies (CMake, Boost, etc.)
RUN apt-get update && apt-get install -y \
    cmake \
    libboost-all-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for Streamlit
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
