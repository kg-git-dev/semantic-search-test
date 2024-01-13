# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the Flask application
EXPOSE 8000

# Command to run your application using gunicorn
CMD ["gunicorn", "--reload", "-b", "0.0.0.0:8000", "app:app"]

