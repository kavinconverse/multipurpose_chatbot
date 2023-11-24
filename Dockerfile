# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /app1

# Copy the current directory all the contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Load environment variables from the .env file
ENV GOOGLE_API_KEY="AIzaSyD3QTNG2zxXusjLaf5O6HjRAFcUVSuFKlg"
ENV ENV_FILE=".env"

# Run streamlit when the container launches
CMD ["streamlit", "run", "frontend.py"]


