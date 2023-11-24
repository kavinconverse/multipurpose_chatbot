#set of instructions to create image
#parent image
FROM python:3.11.6-slim

#directory on image
WORKDIR /app

#copy file from local to image directory'
COPY requirements.txt .

#run the installation file
RUN pip install -r requirements.txt

#copy all file
COPY . .

#create port
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "front.py"]


