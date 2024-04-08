import gdown
import os
def download_files():
    url1 = f"https://drive.google.com/uc?id=16i396nXTpPNilnGELcekzh2SxH63Dpum"
    output1 = "models/pomogranate_grading.h5"
    url2 = f"https://drive.google.com/uc?id=1ZjCXb7YiUswFC6KQYyNr8_DI8pZupVL-"
    output2 = "models/inception_fine_tune.h5"
    if not os.path.exists(output1):    
        print("downloading files")
        gdown.download(url1, output1)
        
    if not os.path.exists(output2):
        gdown.download(url2, output2)
    else:
        print("files already downloaded")


if __name__ == "__main__":
    download_files()