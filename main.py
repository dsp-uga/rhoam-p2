import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
import numpy as np
from matplotlib import pyplot
from PIL import Image
import io
from matplotlib.pyplot import imshow
import tensorflow as tf 
from tensorflow.keras.layers import InputLayer, Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, BatchNormalization, Activation, Conv2D, UpSampling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.optimizers import Adam, SGD

print("Done")

# init spark
spark = SparkSession.builder.appName("P2").getOrCreate()       
sc=spark.sparkContext

# initialize train and test set
train_df = spark.read.csv('gs://uga-dsp/project2/files/X_train.csv', header=True)
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

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

print("Done")




print("Starting Train RDD transfer")
# Convert train data to RDD for manipulation
train_rdd = train_df.rdd
print("RDD transfer done")
train_rdd_new = train_rdd.map(lambda row: rowwise_function(row))

print("RDD map done")

# Convert RDD Back to DataFrame
train_rdd_new.cache()
image_df = train_rdd_new.toDF()

# store crop data into separate RDD
image_df_crop_data = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Sex (subj)').rdd
image_df_crop_data = image_df_crop_data.map(lambda x: (str(x[0]), (int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) ))

# store file names to loop through
image_df = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
valid_list = image_df['Image File'].to_list()
new_list = []
count = 1

# linear loop through file names :(
# analyze in chunks of 500
for chunk in chunker(valid_list,500):
    try:
        image_data = spark.read.format("image").load(chunk).select('image.origin','image.data', 'image.width', 'image.height').rdd.collect()
        new_list.append(chunk)
    except:
        found_bug = 0
        for y in chunk:
            try:
                y2 = spark.read.format("image").load(y).select('image.origin','image.data', 'image.width', 'image.height').rdd.collect()
                # Image.frombytes("RGB", (int(y2[0][2]), int(y2[0][3])), io.BytesIO(y2[0][1]).getvalue())
                new_list.append(y)
                print(f'Found bug: {found_bug}')
                found_bug+=1
            except:
                print(y)
                new_list.append(chunk[-1])
                found_bug+=1
    print(count)
    count += 1
    
# append files that can be read into new list
new_list = sc.parallelize(new_list).flatMap(lambda x: x).collect()
# load into training DF
image_data = spark.read.format("image").load(new_list).select('image.origin','image.data', 'image.width', 'image.height')
image_data = image_data.rdd
print("1")
image_data = image_data.map(lambda x: (str(x[0]), Image.frombytes("RGB", (int(x[2]), int(x[3])), io.BytesIO(x[1]).getvalue())))
print("2")
image_data = image_data.map(lambda x: (x[0], np.asarray(x[1])))
print("3")
image_data = image_data.map(lambda x: (x[0], x[1][:,:,::-1]))
print("4")
image_data = image_data.map(lambda x: (x[0],Image.fromarray(np.uint8(x[1]))))
print("5")
image_data = image_data.leftOuterJoin(image_df_crop_data)
print("6")
image_data = image_data.values()
print("7")

image_data = image_data.map(lambda x: (x[0].crop((int(x[1][0]), int(x[1][1]), int(x[1][2]), int(x[1][3]))), x[1][4]))
print("8")

image_data = image_data.map(lambda x: (x[0].resize((100,100)), x[1]))
print("9")
train_images_cropped = image_data.map(lambda x: (np.array(x[0]), x[1])).collect()
print("10")
train_images_cropped= np.array(train_images_cropped)
print("11")

# test display of first image
first = train_images_cropped[0]
imshow(first[0])

print("Done")






print("Starting Test RDD transfer")
# Convert test data to RDD for manipulation
train_rdd = test_df.rdd
print("RDD transfer done")
train_rdd_new = train_rdd.map(lambda row: rowwise_function(row))

print("RDD map done")

# Convert RDD Back to DataFrame
train_rdd_new.cache()
image_df = train_rdd_new.toDF()

image_df_crop_data = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Sex (subj)').rdd
image_df_crop_data = image_df_crop_data.map(lambda x: (str(x[0]), (int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5])) ))

image_df = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
valid_list = image_df['Image File'].to_list()
new_list = []
count = 1

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

for chunk in chunker(valid_list,500):
    try:
        image_data = spark.read.format("image").load(chunk).select('image.origin','image.data', 'image.width', 'image.height').rdd.collect()
        new_list.append(chunk)
    except:
        found_bug = 0
        for y in chunk:
            try:
                y2 = spark.read.format("image").load(y).select('image.origin','image.data', 'image.width', 'image.height').rdd.collect()
                # Image.frombytes("RGB", (int(y2[0][2]), int(y2[0][3])), io.BytesIO(y2[0][1]).getvalue())
                new_list.append(y)
                print(f'Found bug: {found_bug}')
                found_bug+=1
            except:
                print(y)
                new_list.append(chunk[-1])
                found_bug+=1
    print(count)
    count += 1
    
new_list = sc.parallelize(new_list).flatMap(lambda x: x).collect()
image_data = spark.read.format("image").load(new_list).select('image.origin','image.data', 'image.width', 'image.height')
image_data = image_data.rdd
print("1")
image_data = image_data.map(lambda x: (str(x[0]), Image.frombytes("RGB", (int(x[2]), int(x[3])), io.BytesIO(x[1]).getvalue())))
print("2")
image_data = image_data.map(lambda x: (x[0], np.asarray(x[1])))
print("3")
image_data = image_data.map(lambda x: (x[0], x[1][:,:,::-1]))
print("4")
image_data = image_data.map(lambda x: (x[0],Image.fromarray(np.uint8(x[1]))))
print("5")
image_data = image_data.leftOuterJoin(image_df_crop_data)
print("6")
image_data = image_data.values()
print("7")

image_data = image_data.map(lambda x: (x[0].crop((int(x[1][0]), int(x[1][1]), int(x[1][2]), int(x[1][3]))), x[1][4]))
print("8")

image_data = image_data.map(lambda x: (x[0].resize((100,100)), x[1]))
print("9")
train_images_cropped = image_data.map(lambda x: (np.array(x[0]), x[1])).collect()
print("10")
test_images_cropped= np.array(train_images_cropped)
print("11")
first = train_images_cropped[0]
imshow(first[0])

print("Done")




# split labels and images
x, y = zip(*train_images_cropped)
test_x, test_y = zip(*test_images_cropped)
x = np.array(x)
test_x = np.array(test_x)
y = np.array(y)
test_y = np.array(test_y)
print("Done")



# model initialization
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
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
def train_model(model, x, y):
    print('Training')
    model.fit(x, y, batch_size = 128, epochs = 25, verbose = 1)
    return model

# test model
def test_model(model, x, y):
    evaluation = model.evaluate(x,y)
    print(evaluation)

# test model on large dataset
def final_test(model, test_images):
    output = model.predict(test_images)
    output = sc.parallelize(output).collect()
    print(output[0][0])
    output2 = [np.round_(x).astype(int) for x in output]
    print(output2[0])
    output2 = sc.parallelize(output2)
    # save to text file
    output2.saveAsTextFile('gs://p2_group_bucket1/output.txt')



# initialize large test set
final_test_df = spark.read.csv('gs://uga-dsp/project2/files/Xa_test.csv', header=True)
final_test_rdd = final_test_df.rdd
final_test_rdd_new = final_test_rdd.map(lambda row: rowwise_function(row))

print("Starting Final Test RDD transfer")
# Convert test data to RDD for manipulation
train_rdd = final_test_rdd_new.rdd
print("RDD transfer done")
train_rdd_new = train_rdd.map(lambda row: rowwise_function(row))

print("RDD map done")

# Convert RDD Back to DataFrame
train_rdd_new.cache()
image_df = train_rdd_new.toDF()

image_df_crop_data = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)').rdd
image_df_crop_data = image_df_crop_data.map(lambda x: (str(x[0]), (int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[4])) ))

image_df = image_df.select('Image File', 'X (top left)','Y (top left)', 'X (bottom right)', 'Y (bottom right)', 'Image Width', 'Image Height').toPandas()
valid_list = image_df['Image File'].to_list()
new_list = []
count = 1

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

for chunk in chunker(valid_list,500):
    try:
        image_data = spark.read.format("image").load(chunk).select('image.origin','image.data', 'image.width', 'image.height').rdd.collect()
        new_list.append(chunk)
    except:
        found_bug = 0
        for y in chunk:
            try:
                y2 = spark.read.format("image").load(y).select('image.origin','image.data', 'image.width', 'image.height').rdd.collect()
                # Image.frombytes("RGB", (int(y2[0][2]), int(y2[0][3])), io.BytesIO(y2[0][1]).getvalue())
                new_list.append(y)
                print(f'Found bug: {found_bug}')
                found_bug+=1
            except:
                print(y)
                new_list.append(chunk[-1])
                found_bug+=1
    print(count)
    count += 1
    
new_list = sc.parallelize(new_list).flatMap(lambda x: x).collect()
image_data = spark.read.format("image").load(new_list).select('image.origin','image.data', 'image.width', 'image.height')
image_data = image_data.rdd
print("1")
image_data = image_data.map(lambda x: (str(x[0]), Image.frombytes("RGB", (int(x[2]), int(x[3])), io.BytesIO(x[1]).getvalue())))
print("2")
image_data = image_data.map(lambda x: (x[0], np.asarray(x[1])))
print("3")
image_data = image_data.map(lambda x: (x[0], x[1][:,:,::-1]))
print("4")
image_data = image_data.map(lambda x: (x[0],Image.fromarray(np.uint8(x[1]))))
print("5")
image_data = image_data.leftOuterJoin(image_df_crop_data)
print("6")
image_data = image_data.values()
print("7")

image_data = image_data.map(lambda x: (x[0].crop((int(x[1][0]), int(x[1][1]), int(x[1][2]), int(x[1][3]))), x[1][4]))
print("8")

image_data = image_data.map(lambda x: (x[0].resize((100,100)), x[1]))
print("9")
train_images_cropped = image_data.map(lambda x: (np.array(x[0]), x[1])).collect()
print("10")
final_test_rdd_new = np.array(train_images_cropped)
print("11")
first = train_images_cropped[0]
imshow(first[0])

print("Done")


# large test set does not have labels, split only images
test_x, _ = zip(*final_test_rdd_new)
test_x = np.array(test_x)
print("Done")

# test on large set
final_test(model, test_x)