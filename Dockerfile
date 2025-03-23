FROM python:3.12

ENV APP_HOME /app

ENV PORT 8080

WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run your setup script
RUN python server_setup.py

# Expose the port your application listens on.  Replace 8080 with your port.
EXPOSE 8080

# Define the command to run when the container starts.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app