from scipy.io import wavfile
from sklearn.svm import SVC
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

import glob
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

dataset = [{'path': path, 'label': path.split('/' )[4][8:14] } for path in glob.glob("/Users/Prajwol/Projects/Dataset/**/*.wav")]
df = pd.DataFrame.from_dict(dataset)

# Add a column to store the data read from each wavfile...   
df['x'] = df['path'].apply(lambda x: wavfile.read(x)[1])
df.head()

#resampling
import librosa    
r, u = librosa.load(df['x'], sr=2000)

#Choosing one of the each samples form each catogery 
normal = df[df['label'] == 'normal' ].sample(1)
murmur = df[df['label'] == 'urmuur' ].sample(1)
extrasystole = df[df['label'] == 'xtrsys' ].sample(1)

# Plot the three samples onto three different figures
plt.figure(1, figsize=(8, 5))
plt.title('normal')
plt.plot(normal['x'].values[0], c='m')

plt.figure(2, figsize=(8, 5))
plt.title('murmur')
plt.plot(murmur['x'].values[0], c='c')

plt.figure(3, figsize=(8, 5))
plt.title('extrasystole')
plt.plot(extrasystole['x'].values[0], c='b')

#make the lenght of all audio files same
max_length = max(df['x'].apply(len))

def repeat_to_length(arr, length):
    result = np.empty((length, ), dtype = np.float32)
    l = len(arr)
    pos = 0
    while pos + l <= length:
        result[pos:pos+l] = arr
        pos += l
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result

df['x'] = df['x'].apply(repeat_to_length, length=max_length)
df.head()

df.to_csv("neurons.csv")

# Collect one sample from each of the three classes and plot their waveforms
normal = df[df['label'] == 'normal' ].sample(1)
murmur = df[df['label'] == 'urmuur' ].sample(1)
extrasystole = df[df['label'] == 'xtrsys' ].sample(1)

plt.figure(1, figsize=(8,5))
plt.plot(normal['x'].values[0], c='b', label='normal', alpha=0.8)
plt.plot(murmur['x'].values[0], c='r', label='murmur', alpha=0.8)
plt.plot(extrasystole['x'].values[0], c='g', label='extrasystole', alpha=0.8)

plt.title('Heartbeat waveforms overlayed onto one another')
plt.legend(loc='lower right')
# plt.savefig('temp.png')

fs = 4000
f_normal, t_normal, Sxx_normal = spectrogram(normal['x'].values[0], 4000)
plt.figure(1, figsize=(8,5))
plt.title('Normal')
plt.pcolormesh(t_normal, f_normal, Sxx_normal, cmap='Spectral')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

f_murmur, t_murmur, Sxx_murmur = spectrogram(murmur['x'].values[0], 4000)
plt.figure(2, figsize=(8, 5))
plt.title('Murmur')
plt.pcolormesh(t_murmur, f_murmur, Sxx_murmur, cmap='Spectral')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

f_extra, t_extra, Sxx_extra = spectrogram(extrasystole['x'].values[0], 2000)
plt.figure(3, figsize=(8, 5))
plt.title('Extrasystole')
plt.pcolormesh(t_extra, f_extra, Sxx_extra, cmap='Spectral')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

# Put the data into numpy arrays.
x = np.stack(df['x'].values, axis=0)
y = np.stack(df['label'].values, axis=0)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test, train_filenames, test_filenames = train_test_split(x, df['label'].values, df['path'].values, test_size=0.30)
print("x_train: {0}, x_test: {1}".format(x_train.shape, x_test.shape))


clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32,), 
                    max_iter=20, verbose=True, activation='relu', solver='adam')
h = clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print("Accuracy %.3f" % accuracy_score(y_test, predictions))




from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print("Precision: %.3f" % precision_score(y_test, predictions, average='weighted'))
print('Recall: %.3f' % recall_score(y_test, predictions, average='weighted'))
print('F1 score: %.3f' % f1_score(y_test, predictions, average='weighted'))

#plot loss curve
loss_values = clf.loss_curve_
plt.figure(figsize=(10,5))
plt.title('Loss Curve')
plt.plot(loss_values)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()


#y_pred=clf.predict(x_test)
s=confusion_matrix(y_test,predictions)

#plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
plot_confusion_matrix(s, figsize=(5,5), class_names=['normal','murmur','extrasystole'])

from mlxtend.plotting import plot_learning_curves
plot_learning_curves(x_train, y_train, x_test, y_test, clf)

# Save the train model
from keras.models import model_from_json
model_json = clf.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
clf.save_weights("model.h5")
print("Saved model to disk")

# save the model to disk
from sklearn.externals import joblib
joblib.dump(clf, "model.pkl")

from sklearn.externals import joblib
filename = 'model.sav'
joblib.dump(clf, filename)

tf.keras.models.save_model(
    clf,
    filepath='/Users/Prajwol/Projects/saved_model',
    overwrite=True,
    include_optimizer=True
)

#loaded_model = joblib.load(filename)
#result = loaded_model.score(x_test, y_test)
#print(result)