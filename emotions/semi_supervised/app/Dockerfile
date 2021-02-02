FROM python:3.8-slim

# Select working directory
WORKDIR /deploy

# Copy packages requirements
COPY requirements.txt ./ 
# Run packages installation
RUN pip install --no-cache-dir -r requirements.txt

# Copy the other files
COPY .. ./

# Select entrypoint
ENTRYPOINT ["gunicorn", "app:app"]