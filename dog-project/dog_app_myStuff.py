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



# ----- DATA AUGMENTATION ----- #
# Failed, because of tooooo many files > do easier example first



### TODO: Define your architecture.
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

""" Architecture
- the pre-trained model is given by keras and implemented in 'extract_bottleneck_features.py'
  - since parameter 'inlcude_top' is set to False, we only have to take care of the FC layer
  - default input size for Inception model is (299, 299, depth) > preprocessing of image is implemented too
- FC layer requires a 1D array as input, therefor a global average pooling layer is taken
- last but not least, the output layer with 133 classes is added
"""
InceptionV3_augmodel = Sequential ()

InceptionV3_augmodel.add (GlobalAveragePooling2D (input_shape=train_InceptionV3.shape[1:]))

InceptionV3_augmodel.add (Dense (units=133, activation='softmax'))

InceptionV3_augmodel.summary()


### Data Augmentation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# we only train rotation and translation invariance
train_datagen = ImageDataGenerator (
    rotation_range=30, # Degree range for random rotations
    width_shift_range=0.2, # Range for random horizontal shifts (fraction of total width)
    height_shift_range=0.2, # Range for random vertical shifts (fraction of total height)
    horizontal_flip=True, # Randomly flip inputs horizontally
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory (
    train_files,  # this is the target directory
    target_size= (224, 224),  # all images will be resized
    batch_size=20,
    class_mode=None)


valid_datagen = ImageDataGenerator ()

valid_generator = valid_datagen.flow_from_directory (
    valid_files,
    target_size=(224, 224),
    batch_size=20,
    class_mode=None)




from keras.callbacks import ModelCheckpoint

cbCheckpointer = ModelCheckpoint (
    filepath='saved_models/weights.best.InceptionV3.dataAug.h5',
    verbose=1,
    save_best_only=True)

InceptionV3_augmodel.fit_generator (
    train_generator,
    steps_per_epoch=2000 // 20,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=800 // 20,
    callbacks=[cbCheckpointer])