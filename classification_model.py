from one_model import OneModel

class TwoModels:
	def train(self, df):
		self.fake_model = OneModel("fake_news_score").train(df)
		self.bait_model = OneModel("click_bait_score").train(df)
		return self

	def classify(self, df):
		df["fake_result"] = self.fake_model.classify(df)
		df["bait_result"] = self.bait_model.classify(df)
		return df