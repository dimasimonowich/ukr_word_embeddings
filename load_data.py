import gdown
import zipfile


if __name__ == "__main__":
    data_url = 'https://drive.google.com/uc?id=1_cT9dio8ldtZ917vP_rXETo3a1JfMPW3'
    data_output_file = 'data.zip'

    gdown.download(data_url, data_output_file, quiet=False)

    with zipfile.ZipFile(data_output_file, "r") as f:
        f.extractall(".")

    # saves_url = 'https://drive.google.com/uc?id=1Lw9A78P3-zcMwTBSbLFGHbNpTvf7pPB6'
    # saves_output_file = 'saves.zip'
    #
    # gdown.download(saves_url, saves_output_file, quiet=False)
    #
    # with zipfile.ZipFile(saves_output_file, "r") as f:
    #     f.extractall("saves")

