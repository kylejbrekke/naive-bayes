from DataProcessor import DataProcessor
from TrainingSetCalculator import TrainingSetCalc
from Evaluation import Evaluation as Evaluate


def main():
	"""
	Initialization point for running the entire experiment.
	Takes no arguments
	Returns no output
	"""

	dp = DataProcessor('../datasets', 'filtered') 	# Create DataProcessor object with a path to the data directory and
													# output directory name.

	dp.clean('breast-cancer-wisconsin.csv')  # Clean breast-cancer-wisconsin data
	dp.discretize('iris.csv', categories=6)
	dp.discretize('glass.csv', categories=10)

	dp.filter('breast-cancer-wisconsin.csv', 2)  # Seperate breast-cancer-wisconsin data classes each into their own csv
	dp.mangle('glass_categorized.csv')
	dp.mangle('house-votes-84.csv')
	dp.mangle('iris_categorized.csv')
	dp.mangle('soybean-small.csv')

	dp.filter('breast-cancer-wisconsin.csv', 2)
	dp.filter('breast-cancer-wisconsin.csv', 4)
	dp.filter('breast-cancer-wisconsin_mangled.csv', 2)
	dp.filter('breast-cancer-wisconsin_mangled.csv', 4)

	dp.discretize('glass.csv', categories=10)  # Discretize glass dataset into 10 catagories

	dp.filter('glass_categorized.csv', 1)  # Seperate glass data classes each into their own csv
	dp.filter('glass_categorized.csv', 2)
	dp.filter('glass_categorized.csv', 3)
	dp.filter('glass_categorized.csv', 5)
	dp.filter('glass_categorized.csv', 6)
	dp.filter('glass_categorized.csv', 7)
	dp.filter('glass_categorized_mangled.csv', 1)
	dp.filter('glass_categorized_mangled.csv', 2)
	dp.filter('glass_categorized_mangled.csv', 3)
	dp.filter('glass_categorized_mangled.csv', 5)
	dp.filter('glass_categorized_mangled.csv', 6)
	dp.filter('glass_categorized_mangled.csv', 7)

	dp.filter('house-votes-84.csv', 'republican')  # Seperate house-votes data classes each into their own csv
	dp.filter('house-votes-84.csv', 'democrat')
	dp.filter('house-votes-84_mangled.csv', 'republican')
	dp.filter('house-votes-84_mangled.csv', 'democrat')

	dp.discretize('iris.csv', categories=6)  # Discretize iris data set into 6 catagories
	dp.filter('iris_categorized.csv', 'Iris-setosa')    # Seperate iris data classes each into their own csv
	dp.filter('iris_categorized.csv', 'Iris-versicolor')
	dp.filter('iris_categorized.csv', 'Iris-virginica')
	dp.filter('iris_categorized_mangled.csv', 'Iris-setosa')
	dp.filter('iris_categorized_mangled.csv', 'Iris-versicolor')
	dp.filter('iris_categorized_mangled.csv', 'Iris-virginica')

	dp.filter('soybean-small.csv', 'D1')  # Seperate soybean-small data classes each into their own csv
	dp.filter('soybean-small.csv', 'D2')
	dp.filter('soybean-small.csv', 'D3')
	dp.filter('soybean-small.csv', 'D4')
	dp.filter('soybean-small_mangled.csv', 'D1')
	dp.filter('soybean-small_mangled.csv', 'D2')
	dp.filter('soybean-small_mangled.csv', 'D3')
	dp.filter('soybean-small_mangled.csv', 'D4')

	print("BREAST CANCER WISCONSIN\n")
	# Create a new TrainingSetCalc object with the breast-cancer-wisconsin data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['breast-cancer-wisconsin_2.csv',
													'breast-cancer-wisconsin_4.csv'])
	bcw_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))

	print("HOUSE VOTES 84\n")
	# Create a new TrainingSetCalc object with the house-votes data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['house-votes-84_democrat.csv',
													'house-votes-84_republican.csv'])
	hv_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))

	print("GLASS\n")
	# Create a new TrainingSetCalc object with the glass-catagorized data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['glass_categorized_1.csv', 'glass_categorized_2.csv',
													'glass_categorized_3.csv', 'glass_categorized_5.csv',
													'glass_categorized_6.csv', 'glass_categorized_7.csv'])
	glass_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))

	print("IRIS\n")
	# Create a new TrainingSetCalc object with the iris-catagorized data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['iris_categorized_Iris-setosa.csv',
													'iris_categorized_Iris-versicolor.csv',
													'iris_categorized_Iris-virginica.csv'])
	iris_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))

	print("SOYBEANS\n")
	# Create a new TrainingSetCalc object with the soybean-small data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['soybean-small_D1.csv', 'soybean-small_D2.csv',
													'soybean-small_D3.csv', 'soybean-small_D4.csv'])
	soybean_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))

	print("BREAST CANCER WISCONSIN (MANGLED)\n")
	# Create a new TrainingSetCalc object with the mangled breast-cancer-wisconsin data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['breast-cancer-wisconsin_mangled_2.csv',
													'breast-cancer-wisconsin_mangled_4.csv'])
	bcw_mangled_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))

	print("HOUSE VOTES 84 (MANGLED)\n")
	# Create a new TrainingSetCalc object with the mangled house-votes data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['house-votes-84_mangled_democrat.csv',
													'house-votes-84_mangled_republican.csv'])
	hv_mangled_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))
	Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))  # Perform 10-fold cross validation on the algorithm using the
														   # mangled house-votes data set and print formatted result

	print("GLASS (MANGLED)\n")
	# Create a new TrainingSetCalc object with the mangled glass-catagorized data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['glass_categorized_mangled_1.csv', 'glass_categorized_mangled_2.csv',
													'glass_categorized_mangled_3.csv', 'glass_categorized_mangled_5.csv',
													'glass_categorized_mangled_6.csv', 'glass_categorized_mangled_7.csv'])
	glass_mangled_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))
	Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))  # Perform 10-fold cross validation on the algorithm using the
														# mangled glass-catagorized data set and print formatted result

	print("IRIS (MANGLED)\n")
	# Create a new TrainingSetCalc object with the mangled iris-catagorized data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['iris_categorized_mangled_Iris-setosa.csv',
													'iris_categorized_mangled_Iris-versicolor.csv',
													'iris_categorized_mangled_Iris-virginica.csv'])
	iris_mangled_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))
	Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))  # Perform 10-fold cross validation on the algorithm using the
														   # iris-catagorized data set and print formatted result

	print("SOYBEANS (MANGLED)\n")
	# Create a new TrainingSetCalc object with the mangled soybean-small data set
	tsc = TrainingSetCalc("../datasets/filtered/", ['soybean-small_mangled_D1.csv', 'soybean-small_mangled_D2.csv',
													'soybean-small_mangled_D3.csv', 'soybean-small_mangled_D4.csv'])
	Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))  # Perform 10-fold cross validation on the algorithm using the
														   # mangled soybean-small data set and print formatted result
	soybean_mangled_results = Evaluate.evaluate(tsc.kFoldTrain(tsc.createSets(10)))

	bcw_diffs = []
	hv_diffs = []
	glass_diffs = []
	iris_diffs = []
	soybean_diffs = []

	for i in range(len(bcw_results)):
		bcw_diffs.append((bcw_results[i][0] - bcw_mangled_results[i][0],
						  bcw_results[i][1] - bcw_mangled_results[i][1],
						  bcw_results[i][2] - bcw_mangled_results[i][2]))

	for i in range(len(hv_results)):
		hv_diffs.append((hv_results[i][0] - hv_mangled_results[i][0],
						 hv_results[i][1] - hv_mangled_results[i][1],
						 hv_results[i][2] - hv_mangled_results[i][2]))

	for i in range(len(glass_results)):
		glass_diffs.append((glass_results[i][0] - glass_mangled_results[i][0],
							glass_results[i][1] - glass_mangled_results[i][1],
							glass_results[i][2] - glass_mangled_results[i][2]))

	for i in range(len(iris_results)):
		iris_diffs.append((iris_results[i][0] - iris_mangled_results[i][0],
						   iris_results[i][1] - iris_mangled_results[i][1],
						   iris_results[i][2] - iris_mangled_results[i][2]))

	for i in range(len(soybean_results)):
		soybean_diffs.append((soybean_results[i][0] - soybean_mangled_results[i][0],
							  soybean_results[i][1] - soybean_mangled_results[i][1],
							  soybean_results[i][2] - soybean_mangled_results[i][2]))

	print("BREAST CANCER DIFFS")
	for i in range(len(bcw_diffs)):
		print(bcw_diffs[i][0], "\t", bcw_diffs[i][1], "\t", bcw_diffs[i][2])
	print("\n\n")

	print("HOUSE VOTE DIFFS")
	for i in range(len(hv_diffs)):
		print(hv_diffs[i][0], "\t", hv_diffs[i][1], "\t", hv_diffs[i][2])
	print("\n\n")

	print("GLASS DIFFS")
	for i in range(len(glass_diffs)):
		print(glass_diffs[i][0], "\t", glass_diffs[i][1], "\t", glass_diffs[i][2])
	print("\n\n")

	print("IRIS DIFFS")
	for i in range(len(iris_diffs)):
		print(iris_diffs[i][0], "\t", iris_diffs[i][1], "\t", iris_diffs[i][2])
	print("\n\n")

	print("SOYBEAN DIFFS")
	for i in range(len(soybean_diffs)):
		print(soybean_diffs[i][0], "\t", soybean_diffs[i][1], "\t", soybean_diffs[i][2])
	print("\n\n")

if __name__ == "__main__":  # Run the main function when started from the command line
	main()
