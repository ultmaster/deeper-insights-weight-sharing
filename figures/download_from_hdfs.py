import logging
import os
import zipfile
from argparse import ArgumentParser
from hdfs import InsecureClient

logging.basicConfig(level=logging.INFO)


def download(keyword):
    client = InsecureClient("http://ip_address", user="username")
    root_dir = "/username/dps"
    for folder in client.list(root_dir):
        if keyword not in folder:
            continue
        os.makedirs(os.path.join("data", folder), exist_ok=True)
        for file in client.list(root_dir + "/" + folder):
            target_path = os.path.join("data", folder, file)
            logging.info("Downloading for {}".format(target_path))
            if os.path.exists(target_path):
                logging.warning("{} already exists!".format(target_path))
                continue
            with open(target_path, "wb") as writer, client.read("{}/{}/{}".format(root_dir, folder, file)) as reader:
                writer.write(reader.read())


def extract(keyword):
    root_dir = "data"
    for folder in os.listdir(root_dir):
        if keyword not in folder:
            continue
        for filename in os.listdir(os.path.join(root_dir, folder)):
            target_path = os.path.join(root_dir, folder, filename)
            if not target_path.endswith(".zip"):
                continue
            logging.info("Extracting zipfile: {}".format(target_path))
            with zipfile.ZipFile(target_path) as zf:
                zf.extractall(os.path.join(root_dir, folder))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("keyword")
    parser.add_argument("--no-download", default=False, action='store_true')
    parser.add_argument("--no-extract", default=False, action='store_true')
    args = parser.parse_args()
    if not args.no_download:
        download(args.keyword)
    if not args.no_extract:
        extract(args.keyword)
