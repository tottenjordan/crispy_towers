import os
import json
import numpy as np
import pickle as pkl
from pprint import pprint
from typing import Dict, Tuple

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_recommenders as tfrs

# gcp
from google.cloud import storage

# this repo
from . import train_utils

sys.path.append("..")
import env_config

storage_client = storage.Client(
    project=PROJECT_ID
)

# ========================================
# user tower
# ========================================
class UserContext_Tower(tf.keras.Model):
    '''
    build sequential model for each feature
    pass outputs to dense/cross layers
    concatentate the outputs
    
    the produced embedding represents the features 
    of the user/context known at query time 
    '''
    def __init__(
        self, 
        layer_sizes, 
        vocab_dict,
        embedding_dim,
        projection_dim,
        seed,
        use_cross_layer,
        use_dropout,
        dropout_rate,
        max_tokens,
        max_context_length,
        max_genre_length
    ):
        super().__init__()
        
        # ========================================
        # user features
        # ========================================
        
        # Feature: user_id
        self.user_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict["user_id"], 
                    mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["user_id"]) + 1, 
                    output_dim=embedding_dim,
                    name="user_id_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="user_id_1d"),
            ], name="user_id_emb_model"
        )
        
        # Feature: user_gender
        self.user_gender_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict["user_gender_vocab"], 
                    mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["user_gender_vocab"]) + 1, 
                    output_dim=embedding_dim,
                    name="user_gender_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="user_id_1d"),
            ], name="user_gender_emb_model"
        )
        
        # Feature: user_age
        self.user_age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(100)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="user_age_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="user_age_1d"),
            ], name="user_age_emb_model"
        )
        
        # Feature: user_occupation_text
        self.user_occ_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict["user_occ_vocab"], 
                    mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["user_occ_vocab"]) + 1, 
                    output_dim=embedding_dim,
                    name="user_occ_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="user_occ_1d"),
            ], name="user_occ_emb_model"
        )
        
        # Feature: user_zip_text
        self.user_zip_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict["user_zip_vocab"], 
                    mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["user_zip_vocab"]) + 1, 
                    output_dim=embedding_dim,
                    name="user_zip_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="user_zip_1d"),
            ], name="user_zip_emb_model"
        )
        
        # ========================================
        # context movie features
        # ========================================
        
        # Feature: context_movie_id
        self.context_mv_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict["movie_id"], 
                    mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["movie_id"]) + 1, 
                    output_dim=embedding_dim,
                    name="context_mv_id_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="context_mv_id_1d"),
            ], name="context_mv_id_emb_model"
        )
        
        # context_movie_rating
        self.context_mv_rating_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization([0.0,1.0,2.0,3.0,4.0,5.0]),
                tf.keras.layers.Embedding(
                    input_dim=6 + 1, 
                    output_dim=embedding_dim,
                    name="context_mv_rating_emb_layer",
                ),
            ], name="context_mv_rating_emb_model"
        )
        
        # context_rating_timestamp
        self.context_rating_ts_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(vocab_dict["timestamp_buckets"].tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["timestamp_buckets"]) + 1, 
                    output_dim=embedding_dim,
                    name="context_rating_ts_emb_layer",
                ),
            ], name="context_rating_ts_emb_model"
        )
        
        # Feature: context_movie_genre
        self.context_mv_genre_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens,
                    ngrams=2, 
                    vocabulary=vocab_dict['movie_genre'],
                    # vocabulary=np.array([vocab_dict['movie_genre']]).flatten(),
                    # output_mode='int',
                    # output_sequence_length=env_config.MAX_GENRE_LENGTH,
                    name="context_mv_genre_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="context_mv_genre_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, env_config.MAX_GENRE_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="context_mv_genre_2d"),
            ], name="context_mv_genre_emb_model"
        )
   
        # Feature: context_movie_year
        self.context_mv_year_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1_000)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="context_mv_year_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="context_mv_year_1d"),
            ], name="context_mv_year_emb_model"
        )

        # Feature: context_movie_title
        self.context_mv_title_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens,
                    ngrams=2, 
                    vocabulary=vocab_dict['movie_title'],
                    # vocabulary=np.array([vocab_dict['movie_title']]).flatten(),
                    # output_mode='int',
                    # output_sequence_length=env_config.MAX_CONTEXT_LENGTH,
                    name="context_mv_title_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="context_mv_title_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, env_config.MAX_CONTEXT_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="context_mv_title_2d"),
            ], name="context_mv_title_emb_model"
        )

        # ========================================
        # dense and cross layers
        # ========================================

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform", 
                name="candidate_cross_layer"
            )
        else:
            self._cross_layer = None

        self.dense_layers = tf.keras.Sequential(name="candidate_dense_layers")
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                )
            )
            if use_dropout:
                self.dense_layers.add(tf.keras.layers.Dropout(dropout_rate))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
                )
            )
          
        # ========================================
        # ADDING L2 NORM AT THE END
        # ========================================
        # self.dense_layers.add(
        #     tf.keras.layers.Lambda(
        #         lambda x: tf.nn.l2_normalize(x, 1, epsilon=1e-12, name="normalize_dense_context")
        #     )
        # )
        # self.dense_layers.add(
        #     tf.keras.layers.Lambda(
        #         lambda x: tf.math.l2_normalize(x,0, epsilon=1e-12, name="normalize_dense_context")
        #     )
        # )
        self.dense_layers.add(
            tf.keras.layers.LayerNormalization(name="normalize_dense_context")
        )
        
    # ========================================
    # call
    # ========================================
    def call(self, data):
        '''
        The call method defines what happens when
        the model is called
        '''
       
        all_embs = tf.concat(
            [
                self.user_id_embedding(data['user_id']),
                self.user_gender_embedding(data['user_gender']),
                self.user_age_embedding(data['user_age']),
                self.user_occ_embedding(data['user_occupation_text']),
                self.user_zip_embedding(data['user_zip_code']),
                self.context_mv_id_embedding(data['context_movie_id']),
                self.context_mv_rating_embedding(data['context_movie_rating']),
                self.context_rating_ts_embedding(data['context_rating_timestamp']),
                self.context_mv_genre_embedding(data['context_movie_genre']),
                self.context_mv_year_embedding(data['context_movie_year']),
                self.context_mv_title_embedding(data['context_movie_title']),
                
            ], axis=1
        )

        # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)
        
# ========================================
# candidate (target) movie tower
# ========================================
class Candidate_Tower(tf.keras.Model):
    def __init__(
        self, 
        layer_sizes, 
        vocab_dict,
        embedding_dim,
        projection_dim,
        seed,
        use_cross_layer,
        use_dropout,
        dropout_rate,
        max_tokens,
        max_context_length,
        max_genre_length
    ):
        super().__init__()
        
        # ========================================
        # Candidate features
        # ========================================

        # Feature: target_movie_id
        self.target_mv_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=vocab_dict["movie_id"], 
                    mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["movie_id"]) + 1, 
                    output_dim=embedding_dim,
                    name="target_mv_id_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="target_mv_id_1d"),
            ], name="target_mv_id_emb_model"
        )

        # Feature: target_movie_rating
        self.target_mv_rating_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization([0.0,1.0,2.0,3.0,4.0,5.0]),
                tf.keras.layers.Embedding(
                    input_dim=6 + 1, 
                    output_dim=embedding_dim,
                    name="target_mv_rating_emb_layer",
                ),
            ], name="target_mv_rating_emb_model"
        )

        # Feature: target_rating_timestamp
        self.target_rating_ts_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(vocab_dict["timestamp_buckets"].tolist()),
                tf.keras.layers.Embedding(
                    input_dim=len(vocab_dict["timestamp_buckets"]) + 1, 
                    output_dim=embedding_dim,
                    name="target_rating_ts_emb_layer",
                ),
            ], name="target_rating_ts_emb_model"
        )

        # Feature: target_movie_genres
        self.target_mv_genre_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens,
                    ngrams=2, 
                    vocabulary=vocab_dict['movie_genre'],
                    # vocabulary=np.array([vocab_dict['movie_genre']]).flatten(),
                    # output_mode='int',
                    # output_sequence_length=env_config.MAX_GENRE_LENGTH,
                    name="target_mv_genre_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="target_mv_genre_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, env_config.MAX_GENRE_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="target_mv_genre_2d"),
            ], name="target_mv_genre_emb_model"
        )

        # Feature: target_movie_year
        self.target_mv_year_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1_000)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="target_mv_year_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="target_mv_year_1d"),
            ], name="target_mv_year_emb_model"
        )

        # Feature: target_movie_title
        # Feature: context_movie_title
        self.target_mv_title_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens,
                    ngrams=2, 
                    vocabulary=vocab_dict['movie_title'],
                    # vocabulary=np.array([vocab_dict['movie_title']]).flatten(),
                    # output_mode='int',
                    # output_sequence_length=env_config.MAX_CONTEXT_LENGTH,
                    name="target_mv_title_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="target_mv_title_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, env_config.MAX_CONTEXT_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="target_mv_title_2d"),
            ], name="target_mv_title_emb_model"
        )

        # ========================================
        # dense and cross layers
        # ========================================

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform", 
                name="user_context_cross_layer"
            )
        else:
            self._cross_layer = None

        self.dense_layers = tf.keras.Sequential(name="user_context_dense_layers")
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                )
            )
            if use_dropout:
                self.dense_layers.add(tf.keras.layers.Dropout(dropout_rate))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
                )
            )

        # ========================================
        # ADDING L2 NORM AT THE END
        # ========================================
        # self.dense_layers.add(
        #     tf.keras.layers.Lambda(
        #         lambda x: tf.nn.l2_normalize(x, 1, epsilon=1e-12, name="normalize_dense_candidate")
        #     )
        # )
        # self.dense_layers.add(
        #     tf.keras.layers.Lambda(
        #         lambda x: tf.math.l2_normalize(x,0, epsilon=1e-12, name="normalize_dense_candidate")
        #     )
        # )
        self.dense_layers.add(
            tf.keras.layers.LayerNormalization(name="normalize_dense_candidate")
        )

    # ========================================
    # call
    # ========================================
    def call(self, data):
        '''
        The call method defines what happens when
        the model is called
        '''
       
        all_embs = tf.concat(
            [
                self.target_mv_id_embedding(data["target_movie_id"]),
                self.target_mv_rating_embedding(data["target_movie_rating"]),
                self.target_rating_ts_embedding(data["target_rating_timestamp"]),
                self.target_mv_genre_embedding(data["target_movie_genres"]),
                self.target_mv_year_embedding(data["target_movie_year"]),
                self.target_mv_title_embedding(data["target_movie_title"]),
                
            ], axis=1
        )

        # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)


# ========================================
# combined model: the two towers
# ========================================
class TheTwoTowers(tfrs.models.Model):

    def __init__(
        self, 
        layer_sizes, 
        vocab_dict, 
        parsed_candidate_dataset,
        embedding_dim,
        projection_dim,
        seed,
        use_cross_layer,
        use_dropout,
        dropout_rate,
        max_tokens,
        compute_batch_metrics=False,
        max_context_length,
        max_genre_length
    ):
        super().__init__()

        self.query_tower = UserContext_Tower(
            layer_sizes=layer_sizes,
            vocab_dict=vocab_dict,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            seed=seed,
            use_cross_layer=use_cross_layer,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            max_tokens=max_tokens,
            max_context_length=max_context_length,
            max_genre_length=max_genre_length
        )

        self.candidate_tower = Candidate_Tower(
            layer_sizes=layer_sizes,
            vocab_dict=vocab_dict,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            seed=seed,
            use_cross_layer=use_cross_layer,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            max_tokens=max_tokens,
            max_context_length=max_context_length,
            max_genre_length=max_genre_length
        )

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidate_dataset
                .batch(128)
                # .cache()
                .map(lambda x: (x['target_movie_id'], self.candidate_tower(x))), 
                ks=(10, 50, 100)),
            batch_metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(10, name='batch_categorical_accuracy_at_10'), 
                tf.keras.metrics.TopKCategoricalAccuracy(50, name='batch_categorical_accuracy_at_50')
            ],
            remove_accidental_hits=False,
            name="two_tower_retreival_task"
        )

    def compute_loss(self, data, training=False):
        
        query_embeddings = self.query_tower(data)
        candidate_embeddings = self.candidate_tower(data)
        
        return self.task(
            query_embeddings, 
            candidate_embeddings, 
            compute_metrics=not training,
            candidate_ids=data['target_movie_id'],
            compute_batch_metrics=True # turn off metrics to save time on training
        )
        