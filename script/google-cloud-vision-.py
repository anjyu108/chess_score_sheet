import argparse
import json

def main(args):
    with open(args.input_json_path, encoding="utf-8") as f:
        data = json.load(f)
    for annotation in data['textAnnotations']:
        print(annotation['description'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_path')
    args = parser.parse_args()
    main(args)