import one_model

class TwoModels:
	fake_model = None
	bait_model = None

	def train(self, df):
		fake_model = 1
		bait_model = 2

	def classify(self, df):
		# using fake
		df["fake_result"] = 1
		# using bate
		df["bate_result"] = 1
		return df