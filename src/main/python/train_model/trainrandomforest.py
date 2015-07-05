import sys
import os

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

from collections import OrderedDict
import functools
import itertools

def parseData(line, arrayLabels):
	splittedLine = line.split(",")
	values = [float(s) for s in splittedLine[4:-1]]
	label = arrayLabels[splittedLine[-1]]
	featuresVector = Vectors.dense(values)
	return LabeledPoint(label, featuresVector)

def getLabelsKeyValue(data):
	arrayKeyValue = {}
	labels = data.map(lambda line: line.strip().split(",")[-1])
	label_counts = labels.countByValue()
	sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
	i = 0
	for label, count in sorted_labels.items():
		arrayKeyValue[label] = i
		i = i + 1
	return arrayKeyValue

if __name__ == "__main__":
	conf = SparkConf().setAppName("RandomForest_Anomaly_Detection")
	sc = SparkContext(conf=conf)
	print "Loading data..."
	rawData = sc.textFile('../../resources/kddcup.data.corrected')

	targetEncoded = getLabelsKeyValue(rawData)
	print (targetEncoded)
	data = rawData.map(functools.partial(parseData, arrayLabels=targetEncoded))
	(trainingData, testData) = data.randomSplit([0.7, 0.3])
	model = RandomForest.trainClassifier(trainingData,
										 numClasses=len(targetEncoded),
										 categoricalFeaturesInfo={},
										 numTrees=5,
										 featureSubsetStrategy="auto",
										 impurity='gini',
										 maxDepth=5,
										 maxBins=32)
	# Evaluate model on test instances and compute test error
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
	print('Test Error = ' + str(testErr))
	print('Learned classification forest model:')
	print(model.toDebugString())
	# Save model
	model.save(sc, "model")