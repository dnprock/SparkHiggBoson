/* SimpleNaiveBayes.scala */
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

object SimpleNaiveBayes {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SimpleNaiveBayes")
    val sc = new SparkContext(conf)

    /*val data = sc.textFile("./sample_tree_data.csv")
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }*/
    
    val data = sc.textFile("./sample_naive_bayes_data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    
    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0)
    val prediction = model.predict(test.map(_.features))

    val predictionAndLabel = prediction.zip(test.map(_.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    
    println("Naive Bayes Accuracy: %s".format(accuracy))
    
    // decision tree
    // Load and parse the data file
    val tree_data = sc.textFile("./sample_tree_data.csv")
    val tree_parsedData = tree_data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }

    // Run training algorithm to build the model
    val maxDepth = 5
    val tree_model = DecisionTree.train(tree_parsedData, Classification, Gini, maxDepth)

    // Evaluate model on training examples and compute training error
    val labelAndPreds = tree_parsedData.map { point =>
      val prediction = tree_model.predict(point.features)
      (point.label, prediction)
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count
    println("Training Error = " + trainErr)
    
    val tree_accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println("Decision Tree Accuracy: %s".format(tree_accuracy))
  }
}
