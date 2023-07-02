import os
import pandas as pd

img_dir = ''
annotation_csv = ''

output_dir = ''


df = pd.read_csv(annotation_csv)

for row in df.iterrows():
    