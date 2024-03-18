import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,BatchNormalization,Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape[0])

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') # because convolutional layers in Keras expect input data to be in 4D tensor format:
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') 
X_train /= 255 #normalizing
X_test /= 255 #normalizing
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1),padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu',padding='valid'))
#model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dense(10, activation='softmax'))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42) #making validation set


model.compile(Adam(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

eval = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))


testLoss, testAcc = model.evaluate(X_test, y_test)
print('Test Accuracy:', testAcc)


plt.plot(eval.history['accuracy'], label='accuracy')
plt.plot(eval.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_pred = np.argmax(model.predict(X_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

error_id = np.where(y_pred != y_true)[0]

plt.figure(figsize=(10, 10))
for i, id in enumerate(error_id[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[id].reshape(28, 28))
    plt.title(f'True: {y_true[id]}\nPredicted: {y_pred[id]}')
    plt.axis('off')
plt.show()
