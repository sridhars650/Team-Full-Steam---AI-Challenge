FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run your setup script
RUN python server_setup.py

# Expose the port your application listens on.  Replace 8080 with your port.
EXPOSE 8080

# Define the command to run when the container starts.
CMD ["python", "main.py"] 