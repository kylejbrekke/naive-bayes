class Evaluation:
    @staticmethod
    def evaluate(test_results):
        """
        build confusion matrix and calculate loss functions, then print formatted output to console
        :param test_results: List of tuples of list of actual values and list of predictions
        :return: return nothing
        """
        predictions = []
        actuals = []
        output = []

        for test_result in test_results:  # for each run in k-fold cross validation
            error = 0
            sum_precision = 0
            sum_recall = 0
            prediction_count = len(test_result[0])

            for dataclass in set(test_result[0]):
                true_positives, false_positives, false_negatives, true_negatives = Evaluation._calcConfusion(
                    test_result[1], test_result[0], dataclass)

                if true_positives + false_positives != 0:
                    sum_precision += true_positives / (true_positives + false_positives)

                if true_positives + false_negatives != 0:
                    sum_recall += true_positives / (true_positives + false_negatives)

                error = (false_negatives + false_positives) / prediction_count

            output.append((sum_precision / prediction_count, sum_recall / prediction_count, error))
            predictions = predictions + test_result[1]  # concatinate predictions from all runs
            actuals = actuals + test_result[0]  # concatinate actual values from all runs

        classes = set(actuals)  # create set of every class in the test
        for dataclass in classes:  # for every class
            true_positives, false_positives, false_negatives, true_negatives = Evaluation._calcConfusion(predictions, actuals, dataclass)  # calculate confusion matrix
            Evaluation._printConfusionMatrix(true_positives, false_positives, false_negatives, true_negatives, dataclass)  # print formatted confusion matrix
            error = (false_positives + false_negatives) / len(predictions)  # Calculate error
            precision = true_positives / (true_positives + false_positives)  # Calculate precision
            recall = true_positives / (true_positives + false_negatives)  # Calculate recall
            print("Loss functions for class " + str(dataclass))  # Print formatted loss functions
            print("Error = %s" % error)
            print("Error = %s" % error)
            print("Precision = %s" % precision)
            print("Recall = %s\n\n" % recall)

        return output


    @staticmethod
    def _calcConfusion(predictions, actuals, dataclass):
        """
        Iterate through every prediction and add up confusion matrix values
        :param predictions: list of all predictions in order
        :param actuals: list of all actual values in order
        :param dataclass: class that confusion matrix is calculated relative to
        :return: return values in confusion matrix
        """
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(predictions)):  # for every prediction/actual pair
            if actuals[i] == dataclass:
                if predictions[i] == actuals[i]:
                    true_positives += 1  # true positives are when the actual == class == prediction
                else:
                    false_negatives += 1  # false negatives are when actual == class != prediction
            else:
                if predictions[i] == actuals[i]:
                    true_negatives += 1  # true negatives are when actual != class == prediction
                else:
                    false_positives += 1  # false positives are when actual == prediction != class
        return true_positives, false_positives, false_negatives, true_negatives

    @staticmethod
    def _printConfusionMatrix(true_positives, false_positives, false_negatives,
                              true_negatives, dataclass):
        """
        Print formatted confusion matrix to console.
        :param true_positives:
        :param false_positives:
        :param false_negatives:
        :param true_negatives:
        :param dataclass: class confusion matrix is relative to
        :return: returns nothing
        """
        print("Confusion matrix for class " + str(dataclass))
        print("        \tPositive\tNegative")
        print("Positive\t%d\t\t%d" % (true_positives, false_positives))
        print("Negative\t%d\t\t%d" % (false_negatives, true_negatives))
