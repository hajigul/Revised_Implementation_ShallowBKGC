from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
import torch
import sys
import os

class ShallowBKGC:
    def __init__(self, *, settings, num_entities, num_relations):
        self.settings = settings

        # ================== Determine Dataset Name ==================
        # Try to detect from command line or default to FB15K
        dataset = "FB15K"
        if len(sys.argv) > 2 and "--dataset" in sys.argv:
            idx = sys.argv.index("--dataset")
            if idx + 1 < len(sys.argv):
                dataset = sys.argv[idx + 1].strip('/').split('/')[-1]
        
        print(f"📊 Using dataset: {dataset}")

        # Dynamic embedding filename
        npy_file = f"{dataset}EntTxtWeights.npy"

        # ================== Input Layers ==================
        input_head = Input(shape=(1,), dtype='int32', name='input_head')
        input_tail = Input(shape=(1,), dtype='int32', name='input_tail')

        # ================== Learnable Entity Embeddings ==================
        embedding_layer = Embedding(input_dim=num_entities, 
                                    output_dim=self.settings['embedding_dim'],
                                    input_length=1, 
                                    activity_regularizer=l2(self.settings['reg']))

        head_embedding_e = embedding_layer(input_head)
        head_embedding_drop = Dropout(self.settings['input_dropout'])(head_embedding_e)
        
        tail_embedding_e = embedding_layer(input_tail)
        tail_embedding_drop = Dropout(self.settings['input_dropout'])(tail_embedding_e)

        # ================== Load Correct BERT Text Embeddings ==================
        print(f"📂 Loading precomputed BERT embeddings: {npy_file} ...")
        
        if not os.path.exists(npy_file):
            print(f"❌ Error: {npy_file} not found!")
            print("Please run: python DateProcess_npy.py FB15K")
            raise FileNotFoundError(f"{npy_file} not found")

        embedding_weights = np.load(npy_file, allow_pickle=True)
        
        print(f"Original embedding shape: {embedding_weights.shape}")

        # Handle numpy object array
        if embedding_weights.dtype == object or (len(embedding_weights) > 0 and isinstance(embedding_weights[0], np.ndarray)):
            # Convert list of arrays to single numpy array
            embedding_weights = np.array([arr for arr in embedding_weights])
        
        # Ensure we have a proper numpy array
        if len(embedding_weights.shape) == 1:
            embedding_weights = np.stack(embedding_weights)
        
        # Take only first 'embedding_dim' dimensions
        embedding_weights_dim = embedding_weights[:, :self.settings['embedding_dim']].astype(np.float32)
        
        print(f"✅ Loaded {embedding_weights_dim.shape[0]} entities × {embedding_weights_dim.shape[1]} dims")

        # Create frozen Embedding layer
        embedding_text_layer = Embedding(input_dim=len(embedding_weights_dim), 
                                         output_dim=self.settings['embedding_dim'])
        embedding_text_layer.build((None,))
        embedding_text_layer.set_weights([embedding_weights_dim])
        embedding_text_layer.trainable = False

        head_embedding_e_text = embedding_text_layer(input_head)
        tail_embedding_e_text = embedding_text_layer(input_tail)

        # Combine learnable and static embeddings
        h_embedding = Average()([head_embedding_e_text, head_embedding_drop])
        t_embedding = Average()([tail_embedding_e_text, tail_embedding_drop])

        # ================== Hidden Layers ==================
        hidden_size = self.settings['embedding_dim'] * self.settings['hidden_width_rate']
        
        h_embedding_dense = Dense(hidden_size,
                                  activity_regularizer=l2(self.settings['reg']))(h_embedding)
        h_embedding_dense_d = Dropout(self.settings['hidden_dropout'])(h_embedding_dense)

        t_embedding_dense = Dense(hidden_size,
                                  activity_regularizer=l2(self.settings['reg']))(t_embedding)
        t_embedding_dense_d = Dropout(self.settings['hidden_dropout'])(t_embedding_dense)

        combined = Average()([h_embedding_dense_d, t_embedding_dense_d])

        final_f = Flatten()(combined)
        final_d_relu = Dense(hidden_size, activation="relu")(final_f)
        final_d = Dense(num_relations, activation="sigmoid")(final_d_relu)

        # ================== Compile Model ==================
        optimizer = Adam(learning_rate=0.001)
        
        self.model = Model(inputs=[input_head, input_tail], outputs=final_d)
        self.model.compile(loss='binary_crossentropy', 
                           optimizer=optimizer, 
                           metrics=['accuracy'])
        
        self.model.summary()

    def fit(self, X, y):
        X_Head = np.array(X[:, 0]).reshape(-1, 1)
        X_Tail = np.array(X[:, 1]).reshape(-1, 1)

        self.model.fit([X_Head, X_Tail], 
                       y, 
                       batch_size=self.settings['batch_size'], 
                       epochs=self.settings['epochs'],
                       use_multiprocessing=True, 
                       verbose=1, 
                       shuffle=True)

    def predict(self, X):
        X_Head = np.array(X[:, 0]).reshape(-1, 1)
        X_Tail = np.array(X[:, 1]).reshape(-1, 1)
        return self.model.predict([X_Head, X_Tail])