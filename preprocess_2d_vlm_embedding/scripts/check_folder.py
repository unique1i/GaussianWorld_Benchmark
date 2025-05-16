import os


def read_expected_folders(file1, file2):
    expected = set()
    for file_path in [file1, file2]:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    expected.add(line)
    return expected


def get_actual_subfolders(directory):
    subfolders = set()
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path):
            subfolders.add(entry)
    return subfolders


def main():
    directory = "/home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_default_fix_xyz_gs/language_features_siglip2"
    file1 = "splits/all_splits_v2.txt"
    file2 = "splits/sem_test_v2.txt"

    expected = read_expected_folders(file1, file2)
    actual = get_actual_subfolders(directory)
    missing = expected - actual

    print("Missing subfolders:")
    for folder in sorted(missing):
        print(folder)

    total_expected = len(expected)
    num_missing = len(missing)

    print(f"\nTotal expected folders: {total_expected}")
    print(f"Number of missing folders: {num_missing}")


if __name__ == "__main__":
    main()
