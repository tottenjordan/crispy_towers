import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# setup
PROJECT_ID = sys.argv[1]
NETWORK = sys.argv[2]
REGION = sys.argv[3]
VERSION = sys.argv[4]
BUCKET_NAME = sys.argv[5]
GCS_SUBFOLDER = sys.argv[6]

# Source data
BQ_DATASET = sys.argv[7]
BQ_TABLE = sys.argv[8]
TABLE_SPEC = f'{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}' # need " : " between project and ds

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
JOB_NAME = f'bq-to-tfr-movielens-candidates-{VERSION}-{TIMESTAMP}'
MAX_WORKERS = '40'
RUNNER = 'DataflowRunner'

# storage
ROOT = f'gs://{BUCKET_NAME}/data/candidate-pipeline/{VERSION}'
DATA_DIR = ROOT + '/data'                            # Location to store data
STATS_DIR = ROOT +'/stats/'                          # Location to store stats 
STAGING_DIR = ROOT + '/job/staging/'                 # Dataflow staging directory on GCP
TEMP_DIR =  ROOT + '/job/temp/'                      # Dataflow temporary directory on GCP
TF_RECORD_DIR = ROOT + '/tf-records/'
CANDIDATE_DIR = ROOT + "/candidates/"

QUERY = f"SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"

# output
NUM_TF_RECORDS = 1 # int(TOTAL_MB_DS) // int(TARGET_SHARD_SIZE_MB)

args = {
    'job_name': JOB_NAME,
    'runner': RUNNER,
    # 'source_query': QUERY,
    # 'bq_source_table': TABLE_SPEC,
    'source_bucket': BUCKET_NAME,
    'source_file_pattern': GCS_SUBFOLDER,
    'network': NETWORK,
    'candidate_sink': CANDIDATE_DIR,
    'num_candidate_tfrecords': NUM_TF_RECORDS,
    'project': PROJECT_ID,
    'region': REGION,
    'staging_location': STAGING_DIR,
    'temp_location': TEMP_DIR,
    'save_main_session': True,
    'setup_file': './setup.py',
}

def main():
    from beam_pipe import candidate_pipe_main
    candidate_pipe_main.run(args)
    
if __name__ == "__main__":   
    main()