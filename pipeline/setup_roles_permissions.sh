BILLING="..."
PROJECT="..."
ACCOUNT="..."
SECRET="test"

gcloud projects create ${PROJECT}
gcloud beta billing projects link ${PROJECT} \
--billing-account=${BILLING}

gcloud services enable secretmanager.googleapis.com \
--project=${PROJECT}

gcloud iam service-accounts create ${ACCOUNT} \
--project=${PROJECT}

EMAIL="${ROBOT}@${PROJECT}.iam.gserviceaccount.com"

gcloud iam service-accounts keys create ${PWD}/${ACCOUNT}.json \
--iam-account=${EMAIL}

# See note: the minimum role that includes the perm to list secrets
gcloud projects add-iam-policy-binding ${PROJECT} \
--member=serviceAccount:${EMAIL} \
--role=roles/secretmanager.viewer

echo "test" > test
gcloud secrets create ${SECRET} \
--data-file="test" \
--project=${PROJECT}

python3 -m venv venv
source venv/bin/activate

python3 -m pip install google-cloud-secret-manager