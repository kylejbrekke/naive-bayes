import os
import sys
import getopt
import pandas
import math
from numpy import random as rand

MINIMUM_FILE_NAME_LENGTH = 5  # Required format *.csv which is at least 5 characters
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
			'v', 'w', 'x', 'y', 'z']


class DataProcessor:

	def __init__(self, path, output_directory=""):
		"""
		Constructor for the DataProcessor class.

		:param path: String indicating the relative path to the folder containing your target data sets
		:param !optional output_directory: target directory (relative to the path) to save output files
		"""
		self.path = os.path.join(os.path.dirname(__file__), path)  # store absolute path using the relative path
		if output_directory == "":  # if there is no output directory provided, it is the same as the path
			self.output_path = self.path
		else:  # if an output directory is provided, set it relative to the current path
			self.output_path = os.path.join(self.path, output_directory)

	@staticmethod
	def validate(target):
		target_length = len(target)

		if target_length < MINIMUM_FILE_NAME_LENGTH:
			print("Cannot process target file %s, as the given file name is too short" % target)
			return False

		if target[target_length - 3: target_length] != "csv":
			print("Cannot process target file %s, because it is not a .csv file" % target)
			return False

		return True

	def filter(self, target, value_to_filter, category="Class"):
		"""
		The filter() method opens a designated CSV file, filters it by a given class-value, then creates a new CSV file
		containing the filtered table.

		:param target: String value indicating the target CSV file to be opened
		:param value_to_filter: Target value to filter into the new table
		:param !optional category: Target column to check for the value_to_filter
		"""
		target_length = len(target)

		if self.validate(target):
			output_path = ("%s/%s_%s.csv" % (self.output_path, target[0:target_length - 4], value_to_filter))
			try:
				dataframe = pandas.read_csv(os.path.join(self.path, target))
				output = dataframe[dataframe[category] == value_to_filter]
				output.to_csv(output_path, index=False)
			except FileNotFoundError:
				print("Target file %s does not exist" % target)

	def clean(self, target, key="?", classifier="Class"):
		"""
		Cleans the target data, dropping missing values

		:param target: .csv file to be imported and shuffled
		:param key: String value, missing values are denoted by the key
		:param classifier: String value, we skip this column when analyzing data for drop potential
		"""
		if self.validate(target):
			target_path = os.path.join(self.path, target)
			dataframe = pandas.read_csv(target_path)

			for column in dataframe.columns:  # check every column in the dataframe
				if column == classifier: continue  # skip if it is classifier
				missing_value_index = (dataframe.loc[dataframe[column] == key]).index

				for index in missing_value_index:  # drop the column if it has missing values
					dataframe.drop([index], axis=0, inplace=True)

			print("Overwriting CSV file at ", target_path)
			dataframe.to_csv(target_path, index=False)

	def mangle(self, target, class_label="Class"):
		"""
		Mangle() takes 10% of the columns and shuffles the information, creating noise for that value.

		:param target: .csv file to be imported and shuffled
		:param class_label: String value, column label of column that will be dropped
		"""
		if self.validate(target):
			target_path = os.path.join(self.path, target)
			dataframe = pandas.read_csv(target_path)
			attributes = dataframe.columns

			if class_label in attributes:  # drop class_label column
				attributes = attributes.drop(class_label)

			column_count = len(attributes)
			shuffle_count = math.ceil(column_count / 10)  # 10% of the columns, rounded up
			to_shuffle = rand.choice(column_count, shuffle_count)  # select columns to be shuffled

			for column in to_shuffle:  # shuffles the data in the columns selected to be shuffled
				dataframe[attributes[column]] = dataframe[attributes[column]].sample(frac=1).reset_index(drop=True)

			file_name = target.split('.')[0]
			output_path = os.path.join(self.path, file_name + "_mangled.csv")
			dataframe.to_csv(output_path, index=False)  # outputs to .csv file

	def discretize(self, target, classifier="Class", categories=4):
		"""
		discretizes target csv file into categories

		:param target: .csv file to be imported and discretized
		:param classifier: String value, the column to be skipped
		:param categories: Integer value, the number of sections to split the data into
		"""
		if self.validate(target):
			target_path = os.path.join(self.path, target)
			dataframe = (pandas.read_csv(target_path)).dropna()

			for column in dataframe.columns:  # iterates through each column
				if column == classifier: continue  # skip the class column
				partition_length = math.floor(len(dataframe[column]) / categories)
				dataframe = dataframe.sort_values(by=column).reset_index(drop=True)

				for i in range(categories):  # split the data into categories number of categories
					if i < categories - 1:  # mark all but the last category
						begin = int(i * partition_length)
						end = int(begin + partition_length - 1)
						dataframe.loc[begin:end, column] = ALPHABET[i]
					else:  # mark the last category
						begin = int(i * partition_length)
						dataframe.loc[begin:, column] = ALPHABET[i]

			file_name = target.split('.')[0] + "_categorized.csv"
			dataframe.to_csv(os.path.join(self.path, file_name), index=False)
