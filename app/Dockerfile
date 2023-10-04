FROM python:3.9.16-slim-bullseye
# Create the working directory
#RUN python -m pip install --upgrade pip
USER root
RUN set -ex && mkdir /app
WORKDIR /app
COPY backend/ .
#COPY doubleit_model.pt .
COPY requirements/prod.txt .
RUN pip install -r prod.txt
EXPOSE 8000
ENTRYPOINT [ "python" ]
CMD ["app.py"]

# Run the web server
# EXPOSE 8000
# ENV PYTHONPATH /repo
# CMD python3 /repo/api/app.py


# ocker exec -u root container_1 chmod 777 .

# docker exec -u root -it mlops  bash                                                               


# cloud functions