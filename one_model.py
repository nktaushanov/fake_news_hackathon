
class OneModel:
	nn_classifier = None
	nn_vectorizer = None
	svm_classifier = None
	svm_vectorizer = None
	other_classifier_if_needed = None
	other_vectorizer_if_needed = None

	def __init__(self, score_column_name):
		self.score_column_name = score_column_name

	def train(self, df):
		# fit all of the above
		print "training done"
		return self

	def classify(self, df):
		print "classification done"
		# return a list of 1's and 3's
		return 3
