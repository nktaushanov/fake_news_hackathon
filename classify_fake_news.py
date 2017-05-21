import argparse
import pickle
import pandas as pd
from classification_model import TwoModels
import csv

def main():
	parser = argparse.ArgumentParser(description='Process some integers.')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--train_csv', type=str, 
                    help='Input data file to train a model with.')
	group.add_argument('--classify_input_csv', type=str, 
                    help='Input data csv file to classify with an already trained model.')

	parser.add_argument('--save_model_bin', type=str, 
                    help='Output bin file to save the trained model. (for training only)')
	parser.add_argument('--load_model_bin', type=str, 
                    help='Input bin file to load an already trained model. (for classification only)')
	parser.add_argument('--classified_output_csv', type=str, 
                    help='Output csv file to save an the classification result. (for classification only)')
	parser.add_argument('--classified_output_xls', type=str, 
                    help='Output excel file to save an the classification result. (for classification only) (optional)')
	args = parser.parse_args()

	if args.train_csv is not None:
		do_training(args)
	else:
		do_classification(args)


def do_training(args):
	print "Starting training mode"
	if args.save_model_bin is None:
		print "--save_model_bin must be specified"
		exit()


	df = pd.read_csv(args.train_csv, encoding='utf8')
	df = fix_missing_values(df)

	model = TwoModels().train(df)
	file = open(args.save_model_bin, 'wb')
	pickle.dump(model, file)


def do_classification(args):
	print "Starting classification mode"
	if args.load_model_bin is None or args.classified_output_csv is None :
		print "--load_model_bin and --classified_output_csv must be specified"
		exit()

	df = pd.read_csv(args.classify_input_csv, encoding='utf8')
	df = fix_missing_values(df)


	pkl_file = open(args.load_model_bin, 'rb')
	model = pickle.load(pkl_file)

	classified_df = model.classify(df)
	classified_df.to_csv(args.classified_output_csv, encoding='utf8', index=False, quoting=csv.QUOTE_NONNUMERIC)
	if args.classified_output_xls is not None:
		classified_df.to_excel(args.classified_output_xls, encoding='utf8', index=False)

def fix_missing_values(df):
	df['is_basestring'] = df['Content'].apply(lambda c: isinstance(c, basestring))
	df.loc[df['is_basestring'] == False, 'Content'] = ""
	df.loc[df['is_basestring'] == False, 'Content Url'] = ""
	df.drop('is_basestring', axis=1, inplace=True)
	return df


if __name__ == "__main__":
    main()
