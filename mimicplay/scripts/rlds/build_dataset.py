if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, required=True, help='Directory with the tfrecord files')
    parser.add_argument('--out', type=str, required=True, help='Directory to store the output')

    args = parser.parse_args()

    builder = tfds.builder('egomimic_dataset', data_dir=args.dataset)
    builder.download_and_prepare(data_dir=args.out)
