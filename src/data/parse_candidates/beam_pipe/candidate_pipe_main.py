import os
import gcsfs
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

# beam / dataflow
import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

# def _int64_feature(value):
#     """
#     Get int64 feature
#     """
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def _string_feature(value):
#     """
#     Returns a bytes_list from a string / byte.
#     """
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=value)

# def _string_array(value):
#     """
#     Returns a bytes_list from a string / byte.
#     """
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value).encode('utf-8')]))

# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

class candidates_to_tfexample(beam.DoFn):
    '''
    convert bigqury rows to tf.examples
    '''
    def __init__(self, mode):
        """
          Initialization
        """
        self.mode = mode
    
    
    def process(self, data):
        """
        Convert BQ row to tf-example
        """
        
        example = tf.train.Example(
            features=tf.train.Features(
                feature = {
                    # context sequence item features
                    "context_movie_id":
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=data["context_movie_id"])),
                    "context_movie_rating":
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=data["context_movie_rating"])),
                    "context_rating_timestamp":
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=data["context_timestamp"])),
                    "context_movie_genre":
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=data["context_movie_genres"])),
                    "context_movie_year":
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=data["context_movie_year"])),
                    "context_movie_title":
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=data["context_movie_title"])),

                    # target/label item features
                    "target_movie_id":
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=data["target_movie_id"])),
                    "target_movie_rating":
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=data["target_movie_rating"])),
                    "target_rating_timestamp":
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=data["target_rating_timestamp"])),
                    "target_movie_genres":
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=data["target_movie_genres"])),
                    "target_movie_year":
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=data["target_movie_year"])),
                    "target_movie_title":
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=data["target_movie_title"])),

                    # # global context user features
                    # "user_id":
                    #     tf.train.Feature(
                    #         bytes_list=tf.train.BytesList(value=data["user_id"])),
                    # "user_gender":
                    #     tf.train.Feature(
                    #         bytes_list=tf.train.BytesList(value=data["user_gender"])),
                    # "user_age":
                    #     tf.train.Feature(
                    #         int64_list=tf.train.Int64List(value=data["user_age"])),
                    # "user_occupation_text":
                    #     tf.train.Feature(
                    #         bytes_list=tf.train.BytesList(value=data["user_occupation_text"])),
                    # "user_zip_code":
                    #     tf.train.Feature(
                    #         bytes_list=tf.train.BytesList(value=data["user_zip_code"])),
                }
            )
        )
        
        yield example

        # apache_beam.io.tfrecordio.ReadFromTFRecord
        # apache_beam.io.tfrecordio.WriteToTFRecord

def run(args):
    '''
    define pipeline config
    '''
    
    # f"{env_config.BUCKET_NAME}", 
    # prefix=f'{env_config.EXAMPLE_GEN_GCS_PATH}/val/', 
    
    # BQ_TABLE = args['bq_source_table']
    
    SOURCE_FILE_PATTERN = args["source_file_pattern"] # TODO (wip)
    
    CANDIDATE_SINK = args['candidate_sink']
    RUNNER = args['runner']
    NUM_TF_RECORDS = args['num_candidate_tfrecords']
    QUERY = args['source_query']
    
    pipeline_options = beam.options.pipeline_options.GoogleCloudOptions(**args)
    print(pipeline_options)
    
    # Convert rows to tf-example
    _to_tf_example = candidates_to_tfexample(mode='candidates')
    
    # Write serialized example to tfrecords
    write_to_tf_record = beam.io.WriteToTFRecord(
        file_path_prefix = CANDIDATE_SINK, 
        file_name_suffix=".tfrecords",
        num_shards=1 #hardcoding due to smaller size
    )
    
    # "gs://jt-towers-v1-hybrid-vertex-bucket/data/movielens/m1m/train/ml1m-006-of-008.tfrecord"
    SOURCE_FILE_PATTERN = "gs://jt-towers-v1-hybrid-vertex-bucket/data/movielens/m1m/*.tfrecord"

    with beam.Pipeline(RUNNER, options=pipeline_options) as pipeline:
        (pipeline 
         # | 'Read from BigQuery' >> beam.io.ReadFromBigQuery(table=BQ_TABLE, flatten_results=True)
         | 'Read from TFRecords' >> beam.io.tfrecordio.ReadFromTFRecord(
             file_pattern=SOURCE_FILE_PATTERN, 
             coder=coders.BytesCoder(),
             # compression_type='auto',
             validate=True
         )
         | 'Convert to tf Example' >> beam.ParDo(_to_tf_example)
         | 'Serialize to String' >> beam.Map(lambda example: example.SerializeToString(deterministic=True))
         | "Write as TFRecords to GCS" >> write_to_tf_record
        )