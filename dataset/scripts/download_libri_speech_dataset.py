import os
import tarfile
import torchaudio


def extract_tar_file(tar_path, extract_to):
    """Extracts a .tar.gz file to the specified directory."""
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"{tar_path} not found!")

    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extraction complete.")


def main():
    # Specify the directory for the dataset
    root_dir = "/storage/kfir/data/LibriSpeech"
    os.makedirs(root_dir, exist_ok=True)

    # Download the train-clean-360 subset
    print("Downloading train-clean-360 dataset...")
    train_dataset = torchaudio.datasets.LIBRISPEECH(
        root=root_dir, url="train-clean-360", download=True
    )
    print("train-clean-360 dataset downloaded successfully.")

    # Download the test-clean subset
    print("Downloading test-clean dataset...")
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        root=root_dir, url="test-clean", download=True
    )
    print("test-clean dataset downloaded successfully.")

    # Extract the tar files
    tar_files = [
        os.path.join(root_dir, "train-clean-360.tar.gz"),
        os.path.join(root_dir, "test-clean.tar.gz")
    ]
    for tar_file in tar_files:
        if os.path.exists(tar_file):
            extract_tar_file(tar_file, root_dir)
        else:
            print(f"{tar_file} not found. Skipping extraction.")


if __name__ == "__main__":
    main()
