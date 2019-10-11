from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 15, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#classifier.save('mymodel.h5')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:/MLAI/Hackathon/IBM-project/PlantVillage/train',
                                                 target_size = (64, 64),
                                                 batch_size = 2,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('D:/MLAI/Hackathon/IBM-project/PlantVillage/test',
                                            target_size = (64, 64),
                                            batch_size = 2,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch =15484 ,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 5154)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('D:/MLAI/Hackathon/IBM-project/PlantVillage/single_pred/tom dis.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
training_set.class_indices
result = classifier.predict(test_image)
result
for i in range(15):
    if result[0][i]==1:
        cnt=i
index=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']
print(index[cnt])