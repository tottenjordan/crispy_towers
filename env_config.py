# TODO: edit these
VERSION= "v1"
PREFIX=f"jt-towers-{VERSION}"
PROJECT_ID="hybrid-vertex"
PROJECT_NUM="934903580331"
LOCATION="us-central1"
BUCKET_NAME=f"{PREFIX}-{PROJECT_ID}-bucket"
BUCKET_URI=f"gs://{BUCKET_NAME}"
VPC_NETWORK_NAME="ucaip-haystack-vpc-network"
VPC_NETWORK_FULL=f"projects/{PROJECT_NUM}/global/networks/{VPC_NETWORK_NAME}"
VERTEX_SA=f"{PROJECT_NUM}-compute@developer.gserviceaccount.com"
EXAMPLE_GEN_GCS_PATH="data/movielens/m1m"
TF_RECORD_PREFIX="ml1m"
MAX_CONTEXT_LENGTH=10
MAX_GENRE_LENGTH=10
VOCAB_FILENAME="vocab_dict.pkl"