import pandas as pd
import argparse

def split_file(input_file, output_path, output_prefix, n):
    # Read the input file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Calculate the number of rows per split
    rows_per_split = len(df) // n

    # Split the DataFrame into n equal parts
    splits = [df[i:i+rows_per_split] for i in range(0, len(df), rows_per_split)]

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save each split as a separate CSV file
    for i, split in enumerate(splits):
        output_file = f"{output_prefix}_{i}.csv"
        split.to_csv(os.path.join(output_path, output_file), index=False)
        print(f"Split {i+1} saved as {output_file}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Split input file into n equal parts")
    parser.add_argument("--input-file", help="Path to the input file")
    parser.add_argument("--output-path", help="Path to the output directory")
    parser.add_argument("--output-prefix", help="Prefix for the output file names")
    parser.add_argument("--number-splits", type=int, help="Number of splits")
    args = parser.parse_args()

    # Call the split_file function with the provided arguments
    split_file(args.input_file, args.output_path, args.output_prefix, args.number_splits)

