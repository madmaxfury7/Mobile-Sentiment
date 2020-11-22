import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy import genfromtxt
from prettytable import PrettyTable
from keras.layers import Flatten
from keras.layers import LSTM
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
import h5py
import utility_functions as uf
from keras.preprocessing import sequence
from keras.layers import Dropout
from nltk.tokenize import RegexpTokenizer
from keras.models import model_from_json
from keras.models import load_model


#Function to load all data required for training model
def load_data_all(data_dir, all_data_path, gloveFile, first_run, load_all):

    # Loading the embeddings for the filtered glove list to be used by our neural network model
    # If load_all is true load all embeddings else use filtered glove
    if load_all == True:
        #print("Loading Enbeddings from glove file")
        weight_matrix, word_idx = uf.load_embeddings(gloveFile)
    else:
        #print("Loading embeddings from filtered glove")
        weight_matrix, word_idx = uf.load_embeddings(filtered_glove_path)

    len1=len(word_idx)
    len2=len(weight_matrix)
    #print("Length of word index",len1)
    #print("Length of weight_matrix",len2)

    # Here creating the test, validation and the training data for training the model
    all_data = uf.read_data(all_data_path)
    train_data, test_data, dev_data = uf.training_data_split(all_data, 0.8, data_dir)

    #Using reset_index() fuction for indexing the data in the dataframe
    train_data = train_data.reset_index()
    dev_data = dev_data.reset_index()
    test_data = test_data.reset_index()

    maxSeqLength, avg_words, sequence_length = uf.maxSeqLen(all_data)
    numClasses = 10

    # loading the Training data matrix by using tf_data_pipeline_nltk function from embeddings file
    train_x = uf.tf_data_pipeline_nltk(train_data, word_idx, weight_matrix, maxSeqLength)
    test_x = uf.tf_data_pipeline_nltk(test_data, word_idx, weight_matrix, maxSeqLength)
    val_x = uf.tf_data_pipeline_nltk(dev_data, word_idx, weight_matrix, maxSeqLength)


    # loading the labels data matrix
    train_y = uf.labels_matrix(train_data)
    val_y = uf.labels_matrix(dev_data)
    test_y = uf.labels_matrix(test_data)


    # Summarizing the size of training data
    print("The Training data shape: ")
    print(train_x.shape)
    print(train_y.shape)

    # Summarizing the number of classes
    print("Classes: ")
    print(np.unique(train_y.shape[1]))

    return train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx

def create_model_rnn(weight_matrix, max_words, EMBEDDING_DIM):

    # creating the neural network model
    model = Sequential()
    #Adding Embedding Layer
    model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
    #Adding Bi-Directional LSTM Layer
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    #Adding Dense layer for ReLu
    model.add(Dense(512, activation='relu'))
    #Dropout Layer
    model.add(Dropout(0.50))
    #Dense Layer with Softmax
    model.add(Dense(10, activation='softmax'))
    #Compiling Model
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    #Printing Model Summary
    print(model.summary())
    print()

    return model

def train_model(model,train_x, train_y, test_x, test_y, val_x, val_y, batch_size, path) :

    # Saving The best model to best_model.hdf5 file
    saveBestModel = keras.callbacks.ModelCheckpoint(path+'/model/best_model.hdf5', monitor='val_acc', verbose=0,
                                                    save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # Using EarlyStopping with patience value 3 and set to run for 25 epochs
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

    # Fit the model
    model.fit(train_x, train_y, batch_size=batch_size, epochs=25,
              validation_data=(val_x, val_y), callbacks=[saveBestModel, earlyStopping])
    # Final evaluation of the model
    score, acc = model.evaluate(test_x, test_y, batch_size=batch_size)
    print('The Test score is:', score)
    print('The Test accuracy is:', acc)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    return model

def live_test(trained_model, data, word_idx):

    live_list = []
    live_list_np = np.zeros((56,1))
    # Splitiing the sentence into its words and remove any punctuations 
    # Using Regualr exprssion tokenizer for the same
    tokenizer = RegexpTokenizer(r'\w+')
    data_sample_list = tokenizer.tokenize(data)

    labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")

    # Indexing for the live stage testing of data
    data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
    data_index_np = np.array(data_index)
    print(data_index_np)
    print()


    # Padding with zeros of length 56 i.e maximum length to make the dimensions equal
    padded_array = np.zeros(56) 
    padded_array[:data_index_np.shape[0]] = data_index_np
    data_index_np_pad = padded_array.astype(int)

    # Appending in live_list
    live_list.append(data_index_np_pad)
    live_list_np = np.asarray(live_list)

    # For visualization
    type(live_list_np)

    # Getting the score of the model
    score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
    # print (score)
    # print()

    single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band

    # weighted score of top 3 bands
    top_3_index = np.argsort(score)[0][-3:]
    top_3_scores = score[0][top_3_index]
    top_3_weights = top_3_scores/np.sum(top_3_scores)
    single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)

    #print (single_score)
    #print()
    return single_score_dot

def main():

    # max no of words in training data
    max_words = 56 
    # batch size for training
    batch_size = 2000 
    # size of the word embeddings
    EMBEDDING_DIM = 100 
    # Flag to determine training mode or live testing mode
    train_flag = False 
    path = '/home/sarthak/Desktop/data-mining'
    if train_flag:
        # creating training, validataion and test data sets
        # loading the dataset
        
        data_dir = path+'/Data'
        all_data_path = path+'/Data/'
        gloveFile = path+'/Data/glove/glove_6B_100d.txt'
        first_run = False
        load_all = True

        train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx = load_data_all(data_dir, all_data_path, gloveFile, first_run, load_all)
        # creating model strucutre
        model = create_model_rnn(weight_matrix, max_words, EMBEDDING_DIM)

        # train the model
        trained_model =train_model(model,train_x, train_y, test_x, test_y, val_x, val_y, batch_size, path)   # run model live


        # serializing the weights to HDF5 
        model.save_weights(path+"/model/best_model.hdf5")
        print("Saved model to disk")

    else:
        gloveFile = path +'/Data/glove/glove_6B_100d.txt'
        weight_matrix, word_idx = uf.load_embeddings(gloveFile)
        weight_path = path +'/model/best_model.hdf5'
        loaded_model = load_model(weight_path)
        loaded_model.summary()
        data_sample = "Great!! it is raining today!!."
        result = live_test(loaded_model,data_sample, word_idx)
        print(data_sample)
        print()
        print("Sentiment score of statement using LSTM-model")
        print()
        print (result)
        print()

        ############# Testing the same statements with NLTK-model #############
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(data_sample)
        print("Sentiment score of statement using NLTK-model")
        print()
        #NLTK gives sentiment score between -1 to 1
        #Hence normalizing sentiment score to a value between 0 to 1 for better comaprison
        print((ss['compound']+1)/2.0)

        #load live_test_data
        my_data = genfromtxt('/home/sarthak/Desktop/data-mining/Data/LiveTest/live_test_data.txt', delimiter = ',',dtype=str)
        #load live_test_data_context
        my_context = genfromtxt('/home/sarthak/Desktop/data-mining/Data/LiveTest/live_test_data_context.txt', delimiter = ',',dtype=str)
        #using prettytable to print result in tabular form
        t=PrettyTable(['Sentence','Context','NLTK_Prediction','LSTM_Prediction'])
        x=0
        for data in my_data:
            #print(data)
            result_lstm = live_test(loaded_model,data, word_idx)
            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(data)
            result_nltk=(ss['compound']+1)/2.0
            t.add_row([data,my_context[x],str(result_nltk),str(result_lstm)])
            x+=1
        print(t)
#Running the main function
#Running with train_flag true means that we are in training mode
#Running with train_flag false means that we are in live testing mode
main()
