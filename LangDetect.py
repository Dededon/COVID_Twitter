import os

import pandas as pd
import numpy as np
from langdetect import detect

def checkLang(data_list):
    bool_list = np.empty([len(data_list)], dtype=bool)
    for i in range(len(data_list)):
        if str(data_list[i]) == 'nan':
            bool_list[i] = True
        try:
            if str(data_list[i]) != 'nan':
                if detect(str(data_list[i])) == 'en':
                    bool_list[i] = True
        except Exception as e:
            print(e)
            bool_list[i] = False

    return bool_list


def process(source, target):
    sourceData = pd.read_csv(source, dtype=str, error_bad_lines=False)
    is_text_en = checkLang(sourceData['text'])
    is_quoted_en = checkLang(sourceData['quoted_text'])
    eng_data = sourceData[is_quoted_en & is_text_en]
    eng_data.to_csv(target, index=False)

if __name__ == "__main__":
    source_path = "C:\\Users\\aruba\\Documents\\COMP4641_COVID\\data\\tweets"

    for f in os.listdir(source_path):
        if not f.endswith('.csv'):
            continue
        domain = os.path.abspath(source_path)
        csv_file = os.path.join(domain, f)
        filename = os.path.basename(csv_file).split('.')[0]
        print("Processing "+csv_file)
        output_dest = source_path + '/Eng/'
        if not os.path.exists(output_dest):
            os.mkdir(output_dest)
        output_dest = output_dest + filename + ".csv"
        process(csv_file, output_dest)





