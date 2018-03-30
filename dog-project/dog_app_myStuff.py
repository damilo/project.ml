# preprocessing
from keras.utils import np_utils

# take random pics of human and dog, create a training set
len_set = 400
random_set = []
for i in range (len_set//2):
    random_set.append (random.choice (human_files))
    random_set.append (random.choice (train_files))

X_list = []
for i in range (len_set):
    X_list.append (cv2.resize (cv2.imread (random_set[i], flags=0), (200, 200)))

X_set = np.array (X_list)

y_set = [(x+1)%2 for x in range (len_set)]
y_set = np_utils.to_categorical(y_set)

X_train_set = X_set [:int(len_set*0.8)]
y_train_set = y_set [:int(len_set*0.8)]

X_test_set = X_set [int(len_set*0.8):]
y_test_set = y_set [int(len_set*0.8):]

print ('Training set:')
print ('X: {0} samples'.format (len (X_train_set)))
print ('X shape: {0}'.format (X_train_set.shape))
print ('y: {0} samples'.format (len (y_train_set)))
print ('y shape: {0}'.format (y_train_set.shape))

print ('\nTesting set:')
print ('X: {0} samples'.format (len (X_test_set)))
print ('X shape: {0}'.format (X_test_set.shape))
print ('y: {0} samples'.format (len (y_test_set)))
print ('y shape: {0}'.format (y_test_set.shape))


# do the architecture of MLP
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

q2_model = Sequential ()

q2_model.add (
    Flatten (input_shape=X_train_set.shape[1:]))
q2_model.add (
    Dropout (0.5))
q2_model.add (
    Dense (units=400, activation='relu'))
q2_model.add (
    Dropout (0.2))
q2_model.add (
    Dense (units=200, activation='relu'))
q2_model.add (
    Dropout (0.2))
q2_model.add (
    Dense (units=50, activation='relu'))
q2_model.add (
    Dense (units=2, activation='softmax'))

q2_model.summary ()


q2_model.compile (
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


q2_model.fit (X_train_set, y_train_set, epochs=10, batch_size=20)


score = q2_model.evaluate (X_test_set, y_test_set, batch_size=5)

print ('acc: ', score[1]*100, '%')