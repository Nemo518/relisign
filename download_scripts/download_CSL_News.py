# Modified from https://github.com/NJU-PCALab/OpenVid-1M/blob/main/download_scripts/download_OpenVid.py
import os
import subprocess
import argparse

def download_files(output_directory, download_pose):
    RGB_zip_folder = os.path.join(output_directory, "RGB_download")
    video_folder = os.path.join(output_directory, "rgb_format")
    os.makedirs(RGB_zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    RGB_error_log_path = os.path.join(RGB_zip_folder, "download_log.txt")
    
    # Download RGB format
    for i in range(1, 437):
        url = f"https://huggingface.co/datasets/ZechengLi19/CSL-News/resolve/main/archive_{i:03d}.zip"
        file_path = os.path.join(RGB_zip_folder, f"archive_{i}.zip")
        if os.path.exists(file_path):
            print(f"file {file_path} exits.")
            continue

        command = ["wget", "-O", file_path, url]
        unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
        try:
            subprocess.run(command, check=True)
            print(f"file {url} saved to {file_path}")
            subprocess.run(unzip_command, check=True)
        except subprocess.CalledProcessError as e:
            error_message = f"file {url} download failed: {e}\n"
            print(error_message)
            with open(RGB_error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
    
    # Download pose format (Optional)
    if download_pose:
        pose_zip_folder = os.path.join(output_directory, "pose_download")
        pose_folder = os.path.join(output_directory, "pose_format")
        os.makedirs(pose_zip_folder, exist_ok=True)
        os.makedirs(pose_folder, exist_ok=True)

        pose_error_log_path = os.path.join(pose_zip_folder, "download_log.txt")
        
        for i in range(1, 47):
            url = f"https://huggingface.co/datasets/ZechengLi19/CSL-News_pose/resolve/main/archive_{i:03d}.zip"
            file_path = os.path.join(pose_zip_folder, f"archive_{i}.zip")
            if os.path.exists(file_path):
                print(f"file {file_path} exits.")
                continue

            command = ["wget", "-O", file_path, url]
            unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
            try:
                subprocess.run(command, check=True)
                print(f"file {url} saved to {file_path}")
                subprocess.run(unzip_command, check=True)
            except subprocess.CalledProcessError as e:
                error_message = f"file {url} download failed: {e}\n"
                print(error_message)
                with open(pose_error_log_path, "a") as error_log_file:
                    error_log_file.write(error_message)
        
    # download label
    data_folder = os.path.join(output_directory, "data", "train")
    os.makedirs(data_folder, exist_ok=True)
    data_urls = [
        "https://huggingface.co/datasets/ZechengLi19/CSL-News/resolve/main/data/train/CSL_News_Labels.json"
    ]
    for data_url in data_urls:
        data_path = os.path.join(data_folder, os.path.basename(data_url))
        command = ["wget", "-O", data_path, data_url]
        subprocess.run(command, check=True)

    # delete zip files
    # delete_command = "rm -rf " + RGB_zip_folder
    # os.system(delete_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--output_directory', type=str, help='Path to the dataset directory', default="/path/to/dataset")
    parser.add_argument('--download_pose', action='store_true', help='Whether to download pose or not')
    args = parser.parse_args()
    download_files(args.output_directory, args.download_pose)
