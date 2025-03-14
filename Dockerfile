#General Docker File
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


#############
#AWS ECR Docker File
# Use a lightweight base image
FROM python:3.9-slim as base

# Create a temporary build stage for installing dependencies
FROM base as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final image
FROM base
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY . .
CMD ["python", "app.py"]
