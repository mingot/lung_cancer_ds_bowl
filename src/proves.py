import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates a DL model over some patients')
    parser.add_argument('--input_path', help='Path of the preprocessed patient files')
    parser.add_argument('--model', help='DL model to use')
    parser.add_argument('--ouput_file', help='Output CSV with nodules and scores')
    parser.add_argument('--input_csv',  help='Preselected nodules to pass to the DL')
    args = parser.parse_args()

    if args.input_path:
        print "Hay input path!"
