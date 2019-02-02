#!/bin/python3
import sys
import pandas as pd

def describeColumn(df):
    print("Min " + df.head(0).name + ": "  + str(df.min()))
    print("Max " + df.head(0).name + ": "  + str(df.max()))
    print("Mean " + df.head(0).name + ": "  + str(df.mean()))
    print("Median " + df.head(0).name + ": "  + str(df.median()))

def main():
    df_collection = {}
    df_collection[sys.argv[1]] = pd.read_csv(sys.argv[1])
    df_collection[sys.argv[2]] = pd.read_csv(sys.argv[2])
    for key,val in df_collection.items():
        df = val
        print("\nSummary statistics for: " + key)
        print("Class distribution:")
        print(df['class'].value_counts())

        for species in df['class'].unique():
            print("\n" + species)
            sp_df = df[df['class']==species]
            column_names = list(sp_df.columns.values)
            column_names = column_names[:len(column_names)-1]
            for column_name in column_names:
                describeColumn(sp_df[column_name])

if len(sys.argv) != 3:
    print("Usage: python3 " + sys.argv[0] + " trainingData.csv testData.csv")
    exit(1)

main()
