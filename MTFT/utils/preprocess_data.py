# 전처리된 데이터를 통합하거나 추가적인 전처리 작업을 수행하는 스크립트입니다.

import os
import glob

# 1. 전처리된 데이터를 불러와 하나의 파일로 통합
input_folder = "path_to_preprocessed_data_folder"  # 전처리된 데이터가 있는 폴더
output_file = "path_to_your_project_directory/data/preprocessed_data.txt"  # 통합된 결과를 저장할 경로

# 2. 전처리된 데이터 파일들을 모두 불러오기
files = glob.glob(os.path.join(input_folder, "*.txt"))

# 3. 데이터를 통합하여 하나의 파일에 저장
with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in files:
        with open(fname, encoding="utf-8") as infile:
            outfile.write(infile.read() + "\n")