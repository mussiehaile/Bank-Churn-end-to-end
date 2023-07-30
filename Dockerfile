
# FROM python:3.9-slim

# WORKDIR /code

# COPY ./requirements.txt /code/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# COPY ./src/models/deploy_model.py /code/

# COPY ./model/model.pkl .

# CMD ["uvicorn", "deploy_model:app", "--host", "0.0.0.0", "--port", "80"]


FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src/models/deploy_model.py /code/

# Copy the model.pkl file from the "model" directory to /code/ inside the container
COPY ./model/model.pkl /code/

CMD ["uvicorn", "deploy_model:app", "--host", "0.0.0.0", "--port", "80"]




