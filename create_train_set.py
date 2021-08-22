import argparse
import csv
import pandas as pd

parser = argparse.ArgumentParser(description='Program to convert unanotated sentences into usable input')
parser.add_argument("--data", default="coding_test_sentence_data.txt")
parser.add_argument("--sample_database", default='coding_test_toy_set_acronyms.csv')
parser.add_argument("--test_file_name", default='coding_train_set_acronyms.csv')
args = parser.parse_args()


def main():
    sample_df = pd.read_csv(args.sample_database, sep="|")
    new_df = pd.DataFrame(columns=sample_df.columns)
    
    # Get all unique combinations of acronym and expansion
    sample_data = sample_df.groupby(['acronym', 'expansion', 'type']).size().reset_index().rename(columns={0:'count'})

    with open(args.data) as f:
        data = f.readlines()

    for line in data:
        line = line.lower()
        #Check which of the known expansions occure in the line
        existing_expansions = [expansion.lower() in line for expansion in sample_data['expansion']]
        
        # If one of the known extansions is in the line create a new row in panda df
        if True in existing_expansions:
            index = existing_expansions.index(True)
            row = sample_data.iloc[index]
            line = line.replace(row['expansion'], row['acronym']).strip("\n")
            new_row = {'acronym': row['acronym'], 'expansion':row['expansion'], 'type':row['type'], 'sample':line}
            new_df = new_df.append(new_row, ignore_index=True)

    # write the new panda df to a csv file 
    new_df.to_csv(args.test_file_name, sep="|", mode="w+")


if __name__ == "__main__":
    main()