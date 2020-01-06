import os
import zipfile


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("ALL") and not file.startswith("events") and not file.endswith(".zip") and \
                    not file.endswith(".tar"):
                ziph.write(os.path.join(root, file))
                print(os.path.join(root, file))


if __name__ == '__main__':
    with zipfile.ZipFile('dps-figures-pack.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('analysis', zipf)
        zipdir('data', zipf)
    zipf.close()
