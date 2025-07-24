import pandas as pd

def convert_arff_to_csv(arff_filepath, csv_filepath):

    data_started = False
    data_lines = []
    attributes = []

    with open(arff_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@ATTRIBUTE'):
                # Extract attribute names
                parts = line.split()
                attributes.append(parts[1])
            elif line == '@DATA':
                data_started = True
            elif data_started and line:
                data_lines.append(line.split(','))

    if not data_started:
        raise ValueError("No '@DATA' section found in the ARFF file.")

    df = pd.DataFrame(data_lines, columns=attributes)

    for col in attributes:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    df.to_csv(csv_filepath, index=False)
    return df