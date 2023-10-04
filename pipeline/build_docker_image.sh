# gcloud auth application-default login

gcloud auth configure-docker us-central1-docker.pkg.dev

PROJECT_ID=mlops-400610
DOCKER_IMAGE_URI=${PROJECT_ID}/pipeline-sklearn-iris:v1
docker build ./traincontainer/ -t $DOCKER_IMAGE_URI
DOCKER_TAGGED=us-central1-docker.pkg.dev/${PROJECT_ID}/quickstart-docker-repo/quickstart-image:tag1
docker tag $DOCKER_IMAGE_URI ${DOCKER_TAGGED}
docker push $DOCKER_TAGGED

# f'La uri de la imagen es: {DOCKER_IMAGE_URI}'
## docker build -t mlops .