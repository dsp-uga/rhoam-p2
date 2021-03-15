iimport pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
import numpy as np
from matplotlib import pyplot
from PIL import Image
import io

# from tensorflow.python.lib.io.file_io import FileIO
##CHECK LATER


import tensorflow as tf 
from tensorflow.keras.layers import InputLayer, Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, BatchNormalization, Activation, Conv2D, UpSampling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.optimizers import Adam

print("Done")

# init spark
spark = SparkSession.builder.appName("P2").getOrCreate()       
sc=spark.sparkContext

# initialize train and test set
train_df = spark.read.csv('gs://uga-dsp/project2/files/X_small_train.csv', header=True)
test_df = spark.read.csv('gs://uga-dsp/project2/files/X_small_test.csv', header=True) ## Last column is label 
print("Done")

def rowwise_function(row):
    # convert row to python dictionary:
    row_dict = row.asDict()
    row_dict['Image File'] = 'gs://uga-dsp/project2/images/'+str(row_dict['Image File'])
    # convert dict to row back again:
    newrow = Row(**row_dict)
    # return new row
    return newrow

print("Done")

print("Starting RDD transfer")
# Convert train data to RDD for manipulation
train_rdd = train_df.rdd
print("RDD transfer done")
train_rdd_new = train_rdd.map(lambda row: rowwise_function(row))

print("RDD map done")

# Convert RDD Back to DataFrame
train_rdd_new.cache()
image_df = train_rdd_new.toDF()
image_df = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
print("RDD back to DF")
train_data = []
count = 1
for i, row in image_df.iterrows():
    # Read in images
    image_data = spark.read.format("image").load(row[0]).select('image.data').collect()[0]
    img = Image.frombytes("P", (int(row[5]), int(row[6])), bytes(image_data.data))
    # Crop to just faces
    img = img.crop((int(row[1]), int(row[2]), int(row[3]), int(row[4])))
    img = np.asarray(img)
    # All cropped images in one list
    train_data.append(img)
    print(f'{count} done')
    count +=1
    
print("Done")


print("Starting")
# Convert test data to RDD for manipulation
test_rdd = test_df.rdd
test_rdd_new = test_rdd.map(lambda row: rowwise_function(row))

# Convert RDD Back to DataFrame
test_rdd_new.cache()
test_image_df = test_rdd_new.toDF()
test_image_df = test_image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
test_data = []
for i, row in test_image_df.iterrows():
    # Read in images
    image_data = spark.read.format("image").load(row[0]).select('image.data').collect()[0]
    img = Image.frombytes("P", (int(row[5]), int(row[6])), bytes(image_data.data))
    # Crop to just faces
    img = img.crop((int(row[1]), int(row[2]), int(row[3]), int(row[4])))
    img = np.asarray(img)
    # All cropped images in one list
    test_data.append(img)
    
print("Done")


# Gather train list for tensorflow input
# train_images_cropped = sc.parallelize(train_data).collect()
# train_images_cropped.cache()
for i, x in enumerate(train_data):
    train_data[i] = np.resize(train_data[i], (50, 50, 3))
train_images_cropped = tf.convert_to_tensor(train_data, np.int64)
# train_images_cropped = train_images_cropped.map(lambda x: x.resize(50,50)).collect()
train_labels = np.array(train_df.select('Sex (subj)').rdd.flatMap(lambda x: x).collect())
train_labels = tf.convert_to_tensor(train_labels, np.int64)

# Gather test list for tensorflow input
for i, x in enumerate(test_data):
    test_data[i] = np.resize(test_data[i], (50, 50, 3))
test_images_cropped = tf.convert_to_tensor(test_data, np.int64)
test_labels = np.array(test_df.select('Sex (subj)').rdd.flatMap(lambda x: x).collect())
test_labels = tf.convert_to_tensor(test_labels, np.int64)
print("Done")

model = Sequential()
model.add(Input(shape=(50,50,3)))
model.add(Conv2D(3, kernel_size=5, padding="same"))
model.add(Conv2D(32, kernel_size=5, padding="same"))
model.add(Conv2D(32, kernel_size=5, padding="same"))
model.add(Conv2D(3, kernel_size=5, padding="same"))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam())

def train_model(model, train_images, labels):
    # Combine image data and label to rdd
    print('Training')
    model.fit(train_images, labels, batch_size = 128, epochs = 12, verbose = 1)
    return model

def test_model(model, test_images, labels):
    evaluation = model.evaluate(test_images, labels)
    print(evaluation)

trained_model = train_model(model, train_images_cropped, train_labels)
test_model(trained_model, test_images_cropped, test_labels)