import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import config
import utils
import stream
import model as Model

encoder, features, target = utils.load_data(config.DATA_PATH)
# Create train vectors
train_vectors, count_vectorizer = utils.count_vectorize(features)
utils.make_dir('saved_models')
utils.make_dir('assets')

def train_lstm():
    """Train BiLSTM model for suicide detection."""
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    acc = []
    f1_scores = []
    for i, (train_index, val_index) in enumerate(kfold.split(features, target), 1):
        X_train, y_train = features[train_index], target[train_index]
        X_val, y_val = features[val_index], target[val_index]
        
        tokenizer = text.Tokenizer(num_words=config.VOCAB_SIZE)
        tokenizer.fit_on_texts(list(X_train))
        vocab_size = len(tokenizer.word_counts)
        #print(f'\nVocabulary size: {vocab_size}\n')
        train_tokenized = tokenizer.texts_to_sequences(X_train)
        val_tokenized = tokenizer.texts_to_sequences(X_val)
        X_train = sequence.pad_sequences(train_tokenized, maxlen=config.MAX_SEQ_LEN, dtype='float64')
        X_val = sequence.pad_sequences(val_tokenized,maxlen=config.MAX_SEQ_LEN, dtype='float64')
        
        train_dataset = tf.data.Dataset.from_tensor_slices(
                                                        (X_train, y_train)
                                                        ).batch(config.BATCH_SIZE).repeat()

        val_dataset = tf.data.Dataset.from_tensor_slices(
                                                        (X_val, y_val)
                                                        ).batch(config.BATCH_SIZE)

        model = Model.build_lstm(config)
        early_stopping  = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        mode='min',
                                                        patience=5
                                                        )
        Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"./saved_models/model.h5",
                                                    monitor='val_loss',
                                                    verbose=0,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='min'
                                                    )
        history = model.fit(train_dataset,
                            epochs=config.EPOCHS,
                            steps_per_epoch=280,
                            validation_data=val_dataset,
                            callbacks=[early_stopping, Checkpoint]
                            )
        val_preds = model.predict(X_val)
        val_preds = np.array([int(np.ceil(i)) for i in val_preds.squeeze()])
        f1 = f1_score(y_val, val_preds)
        acc.append(np.mean(history.history['val_accuracy']))
        f1_scores.append(f1)

    print(f"Mean accuracy over 5 folds: {np.round(np.mean(np.array(acc)), 3)}")
    print(f"Mean f1_score over 5 folds: {np.round(np.mean(f1_scores), 3)}")


def train_ensemble(model, features, target):
    """Train suicide prediction model."""
    import pickle

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    acc = []
    f1_scores = []
    for i, (train_index, val_index) in enumerate(kfold.split(features, target), 1):
        X_train, y_train = features[train_index], np.take(target, train_index, axis=0)
        X_val, y_val = features[val_index], np.take(target, val_index, axis=0)

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_preds = np.array([int(np.ceil(i)) for i in val_preds.squeeze()])
        f1 = f1_score(y_val, val_preds)
        print(f'Fold_{i}_f1_score: {f1}')
        f1_scores.append(f1)
    print(f'Mean_F1_score_xgboost: {np.mean(f1_scores)}')
    pickle.dump(model, open('./saved_models/model.pkl', 'wb'))


def predict_on_test(model, _type='tree'):
    """Predict on extracted tweets"""
    test_data = scraper.get_twitter_data()
    test_data = np.array(test_data)
    if _type == 'lstm':
        for i, c in enumerate(test_data, 1):
            test_tokenized = tokenizer.texts_to_sequences(str(c))
            test_tokenized = sequence.pad_sequences(test_tokenized, maxlen=config.MAX_SEQ_LEN, dtype='float64')
            print(f"Tweet: {str(c[0])}:\n Suicidal_Probability: {np.mean(model.predict(test_tokenized))}")
    else:
        for i, c in enumerate(test_data, 1):
            str_to_vec = count_vectorizer.transform(c)
            pred = model.predict_proba(str_to_vec)[:,1]
            print(f"Tweet: {str(c[0])}:\n Suicidal_Probability: {np.round(pred[0], 4)}")


if __name__=='__main__':
    xgb_model = Model.build_ensemble()
    #train_lstm()
    train_ensemble(xgb_model, train_vectors, target)
    predict_on_test(xgb_model)