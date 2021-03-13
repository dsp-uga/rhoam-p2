import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
import numpy as np
from matplotlib import pyplot
from PIL import Image
import io
from google.cloud import storage

# from tensorflow.python.lib.io.file_io import FileIO
##CHECK LATER


# import tensorflow as tf 
# from tensorflow.keras.layers import InputLayer, Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, BatchNormalization, Activation, Conv2D, UpSampling2D, MaxPooling2D
# from tensorflow.keras.models import Sequential, Model, model_from_json
# from tensorflow.keras.optimizers import Adam

# from elephas.utils.rdd_utils import to_simple_rdd
# from elephas.spark_model import SparkModel

# init spark
spark = SparkSession.builder.appName("P2").getOrCreate()       
sc=spark.sparkContext

# initialize train and test set
train_df = spark.read.csv('gs://uga-dsp/project2/files/X_small_train.csv', header=True)
test_df = spark.read.csv('gs://uga-dsp/project2/files/X_small_test.csv', header=True) ## Last column is label 

def rowwise_function(row):
    # convert row to python dictionary:
    row_dict = row.asDict()
    row_dict['Image File'] = 'gs://uga-dsp/project2/images/'+str(row_dict['Image File'])
    # convert dict to row back again:
    newrow = Row(**row_dict)
    # return new row
    return newrow

# Convert to RDD for manipulation
train_rdd = train_df.rdd
train_rdd_new = train_rdd.map(lambda row: rowwise_function(row))

# Convert RDD Back to DataFrame
train_rdd_new.cache()
image_df = train_rdd_new.toDF()
image_df = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
data = []
for i, row in image_df.iterrows():
    # Read in images
    image_data = spark.read.format("image").load(row[0]).select('image.data').collect()[0]
    img = Image.frombytes("P", (int(row[5]), int(row[6])), bytes(image_data.data))
    # Crop to just faces
    img = img.crop((int(row[1]), int(row[2]), int(row[3]), int(row[4])))
    # All cropped images in one liste
    data.append(img)

# Parallelize list for tensorflow input
# cropped_images_rdd = sc.parallelize(data)

##TODO##
# see if there is way to save images to check that they are correct
# add labels rdd or list for elphas implementation

# model = Sequential()
# model.add(Conv2D(3, kernel_size=5, padding="same", input_shape=img_shape))
# model.add(Conv2D(32, kernel_size=5, padding="same"))
# model.add(Conv2D(64, kernel_size=5, padding="same"))
# model.add(Conv2D(32, kernel_size=5, padding="same"))
# model.add(Conv2D(3, kernel_size=5, padding="same"))
# model.add(Activation("tanh"))
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer=Adam())

# def train_model(model, image_data_column, y_column):
#     # Combine image data and label to rdd
#     rdd = to_simple_rdd(sc, image_data_column, y_column)
#     spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
#     spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
