# # ✅ Step 1: Use an official Python base image
# FROM python:3.9

# # ✅ Step 2: Set the working directory inside the container
# WORKDIR /app

# # ✅ Step 3: Copy all project files into the container
# COPY . /app

# # ✅ Step 4: Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # ✅ Step 5: Expose the port Flask runs on
# EXPOSE 5000

# # ✅ Step 6: Define the command to run the app
# CMD ["python", "app.py"]


#Optimized 
#✅ Step 1: Use a lightweight Python base image
FROM python:3.9-slim

# ✅ Step 2: Set the working directory inside the container
WORKDIR /app

# ✅ Step 3: Install system dependencies for MLflow & MySQL connector
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# ✅ Step 4: Copy only necessary files
COPY requirements.txt /app/requirements.txt

# ✅ Step 5: Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ✅ Step 6: Copy the rest of the application files
COPY . /app

# ✅ Step 7: Expose Flask app port
EXPOSE 5000

# ✅ Step 8: Define the command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
