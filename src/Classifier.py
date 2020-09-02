import pandas


class Classifier:
    def __init__(self):
        """
        Constructor function for classifier
        """
        self.probability_table = None
        self.total_class_count = 0  # keeps track of the total number of elements in the training set
        self.total_attributes = 0  # keeps track of the total number of elements

    def classify(self, testing_set):
        """
        Takes in a testing set dataframe, goes through each row and puts the most likely class in a list.
        if for any attribute the response was not seen in the training set it is then added to the dictionary
        using the count zero to calculate the probability.

        :param testing_set: dataframe value
        :return: class_actuals, list of actual values for the testing set.
                 class_predictions, list of predicted values for the testing set.
        """
        class_predictions = []
        class_actuals = []

        for attribute in testing_set.itertuples(index=False):  # iterates through each attribute in the testing set
            class_probabilities = []
            actual_class = None

            for category in self.probability_table.keys():
                total_probability = 1
                category_probability = self.probability_table[category]["Probability"]

                for index in range(testing_set.shape[1]):  # iterates through each index of the rows of the testing set
                    column = testing_set.columns[index]
                    val = attribute[index]

                    if column != "Class" and val in self.probability_table[category]["Attributes"][column]:
                        total_probability *= self.probability_table[category]["Attributes"][column][val]
                    elif column != "Class" and val not in self.probability_table[category]["Attributes"][column]:
                        self.probability_table[category]["Attributes"][column][val] = self._calculateAttributeProbability(0)
                        total_probability *= self.probability_table[category]["Attributes"][column][val]
                    elif column == "Class":
                        actual_class = val

                total_probability = total_probability * category_probability
                class_probabilities.append((category, total_probability))
            classification = None
            highest_probability = 0

            for prob in class_probabilities:
                if prob[1] >= highest_probability:
                    classification = prob[0]
                    highest_probability = prob[1]

            class_predictions.append(classification)
            class_actuals.append(actual_class)

        return class_actuals, class_predictions

    def train(self, training_set):
        """
        function that goes through the nested dict and turns the counts into probabilities sets the global dictionary.

        :param training_set: dataframe used to train
        """
        training_values = self._setup(training_set)

        for _, category in training_values.items():  #
            category["Probability"] = category["Probability"] / self.total_class_count
            attributes = category["Attributes"]

            for attribute, val2 in attributes.items():
                for value in val2.keys():
                    val2[value] = self._calculateAttributeProbability(val2[value])
                    
        self.probability_table = training_values

    def _setup(self, training_set):
        """
        Function that takes the dataframe training set and tallies up the number of each data value for each class and attribute.
        It also creates a dictionary with Class as the main key and a mid key section with Probability and Attributes
        being keys, probability is the number that same class found in the training set.

        :param training_set: dataframe training set used to lay the groundwork for actual training
        :return: list value of tallied classes and attributes based on training_set
        """
        # a function to get the integer position of the Class attribute, stored in j
        class_index = 0
        columns = training_set.columns
        self.total_attributes = len(columns) - 1

        for index in range(len(columns)):
            if columns[index] == "Class":
                class_index = index
                break
        
        # a dictionary to store the the information on which class, which column, which choice, and how many of that
        # choice.
        tally_table = {}
        
        # goes through and tallies up each choice for each attribute, under each class. to be used for calculation
        for attribute in training_set.itertuples(index=False):
            category = attribute[class_index]
            # if type was not in the dict, it adds it to the dict and starts its class counter
            if category not in tally_table:
                tally_table[category] = {"Probability": float(0), "Attributes": {}}
            tally_table[category]["Probability"] += 1  # increments its counter
            self.total_class_count += 1  # increments the total class counter
            
            # goes through the row and adds the attributes to the dict if not already, and adding the value if not
            # already then increments the total num of that value.
            for index in range(len(columns)):
                column = columns[index]

                if column != "Class":
                    if column not in tally_table[category]:
                        tally_table[category]["Attributes"][column] = {}
                    val = attribute[index]

                    if val not in tally_table[category]["Attributes"][column]:
                        tally_table[category]["Attributes"][column][val] = float(0)
                    tally_table[category]["Attributes"][column][val] += 1

        return tally_table

    def _calculateAttributeProbability(self, attr_count):
        """
        helper function to do the calculation for the probability of each attribute response for a certain class

        :param attr_count: int, number of attributes
        :return: float, probability of attribute out of total attributes
        """
        numerator = attr_count + 1
        denominator = self.total_class_count + self.total_attributes
        return numerator / denominator
