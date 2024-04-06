import gdown

url1 = f"https://drive.google.com/uc?id=16i396nXTpPNilnGELcekzh2SxH63Dpum"
output1 = "models/pomogranate_grading.h5"

gdown.download(url1, output1)
