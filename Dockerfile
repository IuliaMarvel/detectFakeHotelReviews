# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR reviews_project

# Copy the contents of the current directory to the working directory in the container
COPY requirements.txt /reviews_project/
# COPY data/ /reviews_project/data/ 
# line above is commented because we will add s3 storage for data later
# this was used to test train and infer modules locally
COPY main_scripts/ /reviews_project/main_scripts/
COPY commands.py /reviews_project/

# Install dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the training script
CMD ["bash"]
