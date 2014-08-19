/* KaggleHiggBoson.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

object KaggleHiggBosonLabel {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("KaggleHiggBosonLabel")
    val sc = new SparkContext(conf)

    val data = sc.textFile("./training-noweight.csv")
    
    val m = Map("s" -> 1, "b" -> 0)
    
    val parsedData = data.map { line =>
      val els = line.split(',')
      // drop first column
      val parts = els.take(0) ++ els.drop(1)
      LabeledPoint(m(parts(parts.size - 1)), Vectors.dense(parts.reverse.tail.reverse.map(_.toDouble)))
    }
    
    val splits = parsedData.randomSplit(Array(0.5, 0.5), seed = 11L)
    val training = splits(0)
    val test_splits = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0)

    val prediction = model.predict(test_splits.map(_.features))
    val predictionAndLabel = prediction.zip(test_splits.map(_.label))
    
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test_splits.count()
    println("Naive Bayes Accuracy: %s".format(accuracy))
    
    predictionAndLabel.coalesce(1).saveAsTextFile("./test-label")
    
    /*val test_data = sc.textFile("./test.csv")
    val test = test_data.map { line =>
      val els = line.split(',')
      val parts = els.take(0) ++ els.drop(1)
      LabeledPoint(1, Vectors.dense(parts.map(_.toDouble)))
    }
    val labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }*/
      
    // Run training algorithm to build the model
    /*val maxDepth = 5
    val tree_model = DecisionTree.train(training, Classification, Gini, maxDepth)

    // Evaluate model on training examples and compute training error
    val labelAndPreds = test.map { point =>
      val prediction = tree_model.predict(point.features)
      (point.label, prediction)
    }*/
    
    /*val data = sc.textFile("./sample_naive_bayes_data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    
    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.4, 0.6), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0)
    val prediction = model.predict(test.map(_.features))

    val predictionAndLabel = prediction.zip(test.map(_.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    
    println("Naive Bayes Accuracy: %s".format(accuracy))
    
    predictionAndLabel.coalesce(1).saveAsTextFile("./test-label")*/
  }
}
