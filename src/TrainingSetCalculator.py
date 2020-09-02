import os
import pandas
import math
from Classifier import Classifier


class TrainingSetCalc:
    def __init__(self, path, targets):
        """
        Constructor for the TrainingSetCalc class.

        :param path: String indicating the relative path to the folder containing your target data sets
        :param targets: List of data file names.
        """
        self.targets = targets
        self.path = os.path.join(os.path.dirname(__file__), path)  # store absolute path using the relative path
        self.output_path = self.path
        target_path = os.path.join(self.path, targets[0])
        self.dataframe = pandas.read_csv(target_path)

    # Initializes and creates a new test set
    def createSets(self, number_of_subsets):
        """
        Create number_of_subsets randomly selected subsets of self.dataframe such that no subsets share data points and
        all data points are in one of the subsets.
        :param number_of_subsets:
        :return subsets: An list of pandas dataframes
        """
        subsets = [pandas.DataFrame(columns=self.dataframe.columns)]*number_of_subsets # Initialize dataframes with
                                                                                       # column names
        for target in self.targets:  # for every csv file provided
            target_path = os.path.join(self.path, target)
            dataframe = pandas.read_csv(target_path).drop_duplicates()
            leng = len(dataframe.index)
            proportion_count = math.floor(leng/number_of_subsets)

            for i in range(number_of_subsets): # for every subset
                sample = dataframe.sample(n=proportion_count, replace=False)
                dataframe = pandas.concat([dataframe, sample]).drop_duplicates(keep=False) # remove sampled entries from
                                                                                           # dataframe
                subsets[i] = subsets[i].append(sample) # Insert sampled entries into subset
            subsets[0] = subsets[0].append(dataframe)  # Append leftover entries to first subset
        return subsets

    def kFoldTrain(self, subsets):
        """
        Perform training on a list of subsets
        :param subsets: list of subsets
        :return test_results: list of tuples each containing a list of classes and list of predicted classes
        """
        number_of_subsets = len(subsets)
        test_results = []

        for i in range(number_of_subsets): # for each subset
            train_set = pandas.DataFrame(columns=self.dataframe.columns) # Initialize train_set with column names

            for j in range(number_of_subsets): # Build training set from all but selected subset
                if i != j:
                    train_set = train_set.append(subsets[j])

            validate_set = subsets[i] # Validation set is only selected set
            test_results.append(self._test(train_set, validate_set))

        return test_results

    @staticmethod
    def _test(train_set, validate_set):
        """
        Train and validate a model.
        :param train_set: Training set
        :param validate_set: Validation set
        :return (class_actuals, class_predictions): tuple of list of of classes and list of predicted classes
        """
        classifier = Classifier()
        classifier.train(train_set)
        class_actuals, class_predictions = classifier.classify(validate_set)
        return (class_actuals, class_predictions)

