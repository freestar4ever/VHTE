FROM bamos/openface:latest
COPY . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
CMD ["python", "web/main.py"]