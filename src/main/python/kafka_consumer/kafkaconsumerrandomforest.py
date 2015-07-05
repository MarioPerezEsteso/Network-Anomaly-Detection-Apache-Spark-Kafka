import sys
import os

# To run this on your local machine, you need to setup kafka and create a producer first:
# http://kafka.apache.org/documentation.html#quickstart

# Path for spark source folder
os.environ['SPARK_HOME'] = "/path/to/spark"

# Append pyspark  to Python Path
sys.path.append("/path/to/spark/python")

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.mllib.tree import RandomForest, RandomForestModel
    from pyspark.mllib.util import MLUtils
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import Vectors

    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

import functools
import itertools
from kafka import KafkaConsumer

def parseData(line):
    splittedLine = line.split(",")
    values = [float(s) for s in splittedLine[4:-1]]
    label = splittedLine[-1]
    featuresVector = Vectors.dense(values)
    return LabeledPoint(label, featuresVector)

if __name__ == "__main__":
    conf = SparkConf().setAppName("RandomForest_Anomaly_Detection_Kafka_Consumer")
    sc = SparkContext(conf=conf)
    savedModel = RandomForestModel.load(sc, "../train_model/model")
    consumer = KafkaConsumer('test', group_id='my_group', bootstrap_servers=['localhost:9092'])
    print("Waiting for messages...")
    for message in consumer:
    	print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition, message.offset, message.key, message.value))
        data = sc.parallelize([message.value])
        testData = data.map(parseData)
        predictions = savedModel.predict(testData.map(lambda x: x.features))
        print("Prediction: ")
        print(predictions.first())