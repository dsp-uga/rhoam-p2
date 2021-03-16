import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
import numpy as np
from matplotlib import pyplot
from PIL import Image
import io
from matplotlib.pyplot import imshow

# from tensorflow.python.lib.io.file_io import FileIO
##CHECK LATER


import tensorflow as tf 
from tensorflow.keras.layers import InputLayer, Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, BatchNormalization, Activation, Conv2D, UpSampling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.optimizers import Adam, SGD

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
# Convert test data to RDD for manipulation
train_rdd = train_df.rdd
print("RDD transfer done")
train_rdd_new = train_rdd.map(lambda row: rowwise_function(row))

print("RDD map done")

# Convert RDD Back to DataFrame
train_rdd_new.cache()
image_df = train_rdd_new.toDF()

image_df_crop_data = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Sex (subj)').rdd
image_df_crop_data = image_df_crop_data.map(lambda x: (str(x[0]), (int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) ))

image_df = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
print("RDD back to DF")
image_data = spark.read.format("image").load(image_df['Image File'].to_list()).select('image.origin','image.data', 'image.width', 'image.height').rdd
image_data = image_data.map(lambda x: (str(x[0]), Image.frombytes("RGB", (int(x[2]), int(x[3])), io.BytesIO(x[1]).getvalue())))
image_data = image_data.map(lambda x: (x[0], np.asarray(x[1])))
image_data = image_data.map(lambda x: (x[0], x[1][:,:,::-1]))
image_data = image_data.map(lambda x: (x[0],Image.fromarray(np.uint8(x[1]))))

image_data = image_data.leftOuterJoin(image_df_crop_data)

image_data = image_data.values()


image_data = image_data.map(lambda x: (x[0].crop((int(x[1][0]), int(x[1][1]), int(x[1][2]), int(x[1][3]))), x[1][4]))


image_data = image_data.map(lambda x: (x[0].resize((100,100)), x[1]))

train_images_cropped = image_data.map(lambda x: (np.array(x[0]), x[1])).collect()
train_images_cropped= np.array(train_images_cropped)

first = train_images_cropped[0]
imshow(first[0])

print("Done")

print("Starting RDD transfer")
# Convert test data to RDD for manipulation
test_rdd = test_df.rdd
print("RDD transfer done")
test_rdd_new = test_rdd.map(lambda row: rowwise_function(row))

print("RDD map done")

# Convert RDD Back to DataFrame
test_rdd_new.cache()
test_image_df = test_rdd_new.toDF()

test_image_df_crop_data = test_image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Sex (subj)').rdd
test_image_df_crop_data = test_image_df_crop_data.map(lambda x: (str(x[0]), (int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) ))

test_image_df = test_image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
print("RDD back to DF")
test_image_data = spark.read.format("image").load(test_image_df['Image File'].to_list()).select('image.origin','image.data', 'image.width', 'image.height').rdd
test_image_data = test_image_data.map(lambda x: (str(x[0]), Image.frombytes("RGB", (int(x[2]), int(x[3])), io.BytesIO(x[1]).getvalue())))
test_image_data = test_image_data.map(lambda x: (x[0], np.asarray(x[1])))
test_image_data = test_image_data.map(lambda x: (x[0], x[1][:,:,::-1]))
test_image_data = test_image_data.map(lambda x: (x[0],Image.fromarray(np.uint8(x[1]))))

test_image_data = test_image_data.leftOuterJoin(test_image_df_crop_data)

test_image_data = test_image_data.values()


test_image_data = test_image_data.map(lambda x: (x[0].crop((int(x[1][0]), int(x[1][1]), int(x[1][2]), int(x[1][3]))), x[1][4]))


test_image_data = test_image_data.map(lambda x: (x[0].resize((100,100)), x[1]))


test_images_cropped = test_image_data.map(lambda x: (np.array(x[0]), x[1])).collect()
test_images_cropped= np.array(test_images_cropped)

first = test_images_cropped[0]
imshow(first[0])

print("Done")

x, y = zip(*train_images_cropped)
test_x, test_y = zip(*test_images_cropped)
x = np.array(x)
test_x = np.array(test_x)
y = np.array(y)
test_y = np.array(test_y)
print("Done")

model = Sequential()
model.add(Input(shape=(100,100, 3)))
model.add(Conv2D(32, kernel_size=5, padding="same"))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(64, kernel_size=5, padding="same"))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(128, kernel_size=5, padding="same"))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

def train_model(model, x, y):
    # Combine image data and label to rdd
    print('Training')
    model.fit(x, y, batch_size = 128, epochs = 25, verbose = 1)
    return model

def test_model(model, x, y):
    evaluation = model.evaluate(x,y)
    print(evaluation)

def final_test(model, test_images):
    output = model.predict(test_images)
    output = sc.parallelize(output)
    output.saveAsTextFile('gs://p2_group_bucket/output.txt')

final_test_df = spark.read.csv('gs://uga-dsp/project2/files/Xa_test.csv', header=True)

final_test_rdd = final_test_df.rdd
final_test_rdd_new = final_test_rdd.map(lambda row: rowwise_function(row))

# Convert RDD Back to DataFrame
final_test_rdd_new.cache()
test_image_df = final_test_rdd_new.toDF()
test_image_df_crop_data = test_image_df.select('X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)').rdd
test_image_df = test_image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
image_data = spark.read.format("image").load(test_image_df['Image File'].to_list()).select('image.origin','image.data', 'image.width', 'image.height').rdd
image_data = image_data.map(lambda x: Image.frombytes("RGB", (int(x[2]), int(x[3])), io.BytesIO(x[1]).getvalue()))
image_data = image_data.map(lambda x: np.asarray(x))
image_data = image_data.map(lambda x: x[:,:,::-1])
image_data = image_data.map(lambda x: Image.fromarray(np.uint8(x)))
                            
image_data = image_data.zipWithIndex().map(lambda x: (x[1], x[0]))
test_image_df_crop_data = test_image_df_crop_data.zipWithIndex().map(lambda x: (x[1], x[0]))
image_data = image_data.leftOuterJoin(test_image_df_crop_data)
image_data = image_data.values()

image_data = image_data.map(lambda x: x[0].crop((int(x[1][0]), int(x[1][1]), int(x[1][2]), int(x[1][3]))))
image_data = image_data.map(lambda x: x.resize((100,100)))
test_images_cropped = image_data.map(lambda x: tf.keras.preprocessing.image.img_to_array(x)).collect()
final_test_images_cropped= np.array(test_images_cropped)
print("Done")

final_test(model, final_test_images_cropped)