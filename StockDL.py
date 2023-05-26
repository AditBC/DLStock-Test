import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import decimal
import keras.backend as K

np.random.seed(10)


df = pd.read_csv("data/^GSPC.csv", index_col =[0])

df.set_index(pd.DatetimeIndex(df["Date"]), inplace=True)

print(df)

data = df.copy()


# data = pd.read_csv("data/^GSPC.csv")

def signalTarget(row):
  if row >= 0:
    target = 1
  else:
    target = 0
  return target

data['LastDiff'] = (data['Close'] - data['Close'].shift(1))/data['Close'].shift(1)*100
# data['LastDiff2'] = data['Close'] - data['Close'].shift(2)/data['Close'].shift(2)*100
# data['LastDiff3'] = data['Close'] - data['Close'].shift(3)/data['Close'].shift(3)*100
data['Strength'] = (data['Close'] - data['Open'])/data['Open']*100
# data['Strength-1'] = data['Strength'].shift(1)
# data['Strength-2'] = data['Strength'].shift(2)
# data['Strength-3'] = data['Strength'].shift(3)
data['RSI']=ta.rsi(data.Close, length=5)
# data['RSITarget'] = data['RSI'].apply(rsiTarget)
# data['SMAF']=ta.ema(data.Close, length=2)
# data['SMAM']=ta.ema(data.Close, length=5)
# data['EMAF']=ta.ema(data.Close, length=2)
# data['EMAM']=ta.ema(data.Close, length=5)
data['EMAS']=ta.ema(data.Close, length=9)
data4 =  ta.stoch(high=data.High, low=data.Low, close=data.Close, k=5, d=3)
# data['MOMF'] = data4.iloc[:,0]
# data['MOMS'] = data4.iloc[:,1]
data['MACD'] = ta.macd(data.Close).iloc[:,0]
data['CHKTRND'] = (data['MACD'] - data['EMAS'])
data['TRND'] = data['CHKTRND'].apply(signalTarget)
# data['EMAS']=ta.ema(data.Close, length=15)
data2 = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=10)
# data['ADX'] = data2.iloc[:,0] 
# data['DMP'] = data2.iloc[:,1]
# data['DMN'] = data2.iloc[:,2] 
data3 = ta.bbands(close=data['Close'], length =5, std=2)
data['BBL'] = data3.iloc[:,0] 
data['BBM'] = data3.iloc[:,1] 
data['BBU'] = data3.iloc[:,2] 
data['BBB'] = data3.iloc[:,3] 
data['BBP'] = data3.iloc[:,4] 
data5 = df.ta.vwap(anchor="D")
data['VWAP'] = data5

del data["Adj Close"]
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
# del data["Open"]
# del data["Close"]
# del data["High"]
# del data["Low"]

data.dropna(inplace = True)
# data.reset_index(inplace = True)
print(data.head())
data.drop(['Date'], axis = 1, inplace = True)
data.tail()

training_set = data.iloc[:,1:len(data.columns)]
print(training_set.head())

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X= []
y= []

backcandles = 3
# print(training_set_scaled.shape[0])
for j in range(len(training_set.columns)-1):
    X.append([])
    for i in range(backcandles, training_set_scaled.shape[0]):#backcandles+2
        X[j].append(training_set_scaled[i-backcandles:i, j])
        

#move axis from 0 to position 2
X=np.moveaxis(X, [0], [2])

#Erase first elements of y because of backcandles to match X length

X, yi =np.array(X), np.array(training_set_scaled[backcandles:,-1])
y=np.reshape(yi,(len(yi),1))
#y=sc.fit_transform(yi)
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# print(X)
# print(X.shape)
# print(y)
# print(y.shape)

splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit].astype(float), X[splitlimit:].astype(float)
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train)
print(X_test)

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+ K.epsilon())
    return f1_val

regressor = Sequential()
# # Adding the first LSTM layer and some Dropout regularisation
# # Note that in input shape in this example is (no of time steps x no of variables) is ( 60 x 1)
# # Number of LSTM units (50) has no relationship to no of time steps.
# # With 50 units this layer produces an output sequence of 50 time steps
regressor.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(50, activation='relu' ,input_shape=(X_train.shape[1], X_train.shape[2])))
# regressor.add(LSTM(150, activation='relu', input_shape=(backcandles, X_train.shape[2])))
regressor.add(Dropout(0.2))
# regressor.add(Flatten())
# # Adding a second LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 100, return_sequences = True))
# regressor.add(Dropout(0.2))
# # Adding a third LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.3))
# # Adding a fourth LSTM layer and some Dropout regularisation
# # This layer produces a single output
# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))
# Adding the output layer
# This is the final layer produces the required numerical prediction
# regressor.add(Dense(y_train.shape[1]))
regressor.add(Dense(1, activation = 'sigmoid'))

# # Compiling the RNN
adam = optimizers.Adam(lr = 0.0001)
nadam = optimizers.Nadam(lr= 0.0002)
# # regressor.compile(optimizer = adam , loss = 'mse')
regressor.compile(optimizer = adam , loss = 'binary_crossentropy', metrics=['accuracy', get_f1])

regressor.summary()
# plot_model(regressor, to_file='model/Lstm_plot.png', show_shapes=True, show_layer_names=True)

history = regressor.fit(x=X_train, y=y_train,  epochs = 300, batch_size = 32, shuffle=False,validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=50)])

scores = regressor.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("F1: %.2f" % (scores[2]))

y_pred = regressor.predict(X_test)
for i in range(20):
    print(y_pred[i], y_test[i])

plt.plot(history.history['loss'], label='Training loss')
plt.xlabel('Epochs')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.ylabel('Loss')
plt.legend()

plt.figure(figsize=(16,8))
plt.plot(y_test, color = 'black', label = 'Test')
plt.xlabel('Days')
plt.plot(y_pred, color = 'green', label = 'pred')
plt.ylabel('Target Class')
plt.legend()
plt.show()

y_pred2 = []
for i in range(len(y_pred)):
  if (y_pred[i]>0.50):
    y_pred2.append(1)
  else:
    y_pred2.append(0)

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# print("F1 Score: ",get_f1(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred2))
print("Accuracy: ", accuracy_score(y_test, y_pred2))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred2).ravel()
print("True Positive: ", tp)
print("False Negative: ", fn)
print("True Negative: ", tn)
print("False Positive: ", fp)