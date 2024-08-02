# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the config.env file to the working directory
COPY config.env /app

# Install dotenv to load environment variables
RUN pip install python-dotenv

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=AgriAdvisor.py

# Run the command to start the app
CMD ["python", "AgriAdvisor.py"]
