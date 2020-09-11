## Author: Stephen Mugisha
## Github: https://github.com/steph-en-m/deep_learning/
## Kaggle-Notebook: https://www.kaggle.com/stephenmugisha/roberta-vs-watson

# An NLI Roberta Model with TensorFlow and use TPU for acceleration.

# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf

from transformers import RobertaTokenizer, XLMRobertaTokenizer, TFXLMRobertaModel, TFRobertaModel
from kaggle_datasets import KaggleDatasets

warnings.filterwarnings('ignore')

os.listdir("../input/contradictory-my-dear-watson") #List files in data dir


# ### TPU configuration

# In[2]:


def TPUSetup():
    """Configure TPU."""
    print('=========Configuring TPU=========')
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('=========Finished TPU configuration=========')
    except ValueError:
        # default to CPU and single GPU if TPU isn't detected
        strategy = tf.distribute.get_strategy()
    print('Number of replicas in sync:', strategy.num_replicas_in_sync)
    return strategy 

strategy = TPUSetup()
BATCH_SIZE= 16 * strategy.num_replicas_in_sync #ensures utilization of all tpu cores for training speedup
AUTO = tf.data.experimental.AUTOTUNE


CONFIG = {
    'train_path':'../input/contradictory-my-dear-watson/train.csv',
    'test_path':'../input/contradictory-my-dear-watson/test.csv',
    'train':{
        'model_name': 'jplu/tf-xlm-roberta-large',
        'batch_size': BATCH_SIZE,
        'epochs': 15,
    }
    
}


def load_data(path:[str])->pd.DataFrame:
    """Loads files in the given file path.
    Default path order: [train_path, test_path]
    """
    train = pd.read_csv(path[0])
    test = pd.read_csv(path[1])
    return train, test

train_df, test_df = load_data([CONFIG['train_path'], CONFIG['test_path']])

# Exploring the shape of the datasets
print(f'Train Data shape {train_df.shape}')
print(f'Test Data shape: {test_df.shape}')



# Building the Model


MAX_LEN = 90
tokenizer = XLMRobertaTokenizer.from_pretrained(CONFIG['train']['model_name'])

def text_encode(sentence):
    """Encodes an input sentence using the bert tokenizer.
    """
    tokens = list(tokenizer.tokenize(sentence))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def roberta_encode(hypotheses, premises, tokenizer):
    """
    Takes input_ids, input_masks, and input_type_ids inputs
    to allow the model to know that the premise and hypothesis are distinct sentences
    and also to ignore any padding from the tokenizer.
    the [CLS] token denotes the beginning of the inputs,
    a [SEP] token denotes the separation between the premise and the hypothesis.
    Padding ensures all of the inputs to be the same size.
    """
    num_examples = len(hypotheses)
    hypothesis = tf.ragged.constant([text_encode(text) for text in np.array(hypotheses)])
    premise = tf.ragged.constant([text_encode(text) for text in np.array(premises)])
    
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * hypothesis.shape[0]
    input_ids = tf.concat([cls, hypothesis, premise], axis=-1)
    input_mask = tf.ones_like(input_ids).to_tensor()
    
    type_cls = tf.zeros_like(cls)
    type_hypothesis = tf.zeros_like(hypothesis)
    type_premise = tf.zeros_like(premise)
    input_type_ids = tf.concat([type_cls, type_hypothesis, type_premise], axis=-1).to_tensor()
    
    inputs = {
        'input_ids':input_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }
    
    return inputs

def build_model(model_name: str):
    """NLI Roberta XLM model"""
    roberta_encoder = TFXLMRobertaModel.from_pretrained(model_name)
    input_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')
    
    embedding = roberta_encoder([input_ids, input_mask, input_type_ids])[0]
    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:, 0, :])
    
    model = tf.keras.Model(inputs=[input_ids, input_mask, input_type_ids], outputs=output)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                 )
    
    return model
    

# Training and test data
train_input = roberta_encode(train_df['hypothesis'].values, train_df['premise'].values, tokenizer)
test_input = roberta_encode(test_df['premise'].values, test_df['hypothesis'].values, tokenizer)

# Learning rate Scheduler
def build_lrfn(lr_start=0.00001, lr_max=0.00003, 
               lr_min=0.000001, lr_rampup_epochs=3, 
               lr_sustain_epochs=0, lr_exp_decay=.6):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn


_lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(_lrfn, verbose=1)
Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"roberta_base.h5",
                                              monitor='val_loss',
                                              verbose=0,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min'
                                             )



# Moving the model to TPU for training
with strategy.scope():
    model = build_model(CONFIG['train']['model_name'])
    model.summary()
    
model.fit(train_input, 
          train_df['label'].values,
          epochs=CONFIG['train']['epochs'],
          verbose = 1,
          batch_size = CONFIG['train']['batch_size'],
          validation_split=0.2,
          callbacks=[Checkpoint]
         )



predictions = [np.argmax(i) for i in model.predict(test_input)]





