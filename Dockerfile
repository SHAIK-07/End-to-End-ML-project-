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
FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt
CMD ["python3", "app.py"]