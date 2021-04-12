import pandas as pd
import numpy as np
import os

def findKeyword(data, keywords, target, threshold):
    k_words = keywords[0].lower()
    for i in range(1, len(keywords)):
        k_words = k_words + '|' + keywords[i].lower()
    text = data[data['text'].fillna('nan').str.contains(k_words)].reset_index(drop=True)
    quoted_text = data[data['quoted_text'].fillna('nan').str.contains(k_words)].reset_index(drop=True)
    text_bool = thresholdCheck(text, threshold, keywords)
    quoted_text_bool = thresholdCheck(quoted_text, threshold, keywords)

    text_res = text[text_bool]
    quote_res = quoted_text[quoted_text_bool]

    result = data[np.logical_or(text_bool, quoted_text_bool)]
    # if len(text_res) == 0:
    #     result = quote_res
    # elif len(quote_res) == 0:
    #     result = text_res
    # else:
    #     result = pd.merge(text_res, quote_res, on='columns', how='outer')
    result.to_csv(target, date_format='%s', index=False)


def thresholdCheck(text_list, threshold, keywords):
    bool_list = np.empty([len(text_list)], dtype=bool)
    if len(text_list) == 0:
        return bool_list
    for i in range(len(text_list)):
        if (keywordCount(text_list[i], keywords) >= threshold):
            bool_list[i] = True
        else:
            bool_list[i] = False
    return bool_list


def keywordCount(text, keywords):
    num = 0
    text = str(text).lower()
    for keyword in keywords:
        if keyword.lower() in text:
            num += 1
    return num


if __name__ == '__main__':
    keyword_txt = 'C:\\Users\\aruba\\Documents\\COMP4641_COVID\\keywords.txt'
    source_path = 'C:\\Users\\aruba\\Documents\\COMP4641_COVID\\data\\tweets\\Eng'
    out_path = 'C:\\Users\\aruba\\Documents\\COMP4641_COVID\\output'
    # keywords = list()
    # with open(keyword_txt) as file:
    #     for line in file:
    #         keywords = line.strip().split(',')
    keywords = ["vaccine","BioNTech","Pfizer","Moderna","dose","CureVac","inject","inoculate","Gates","qanon"
        ,"infertility","mRNA","DNA","microchip","chip","Sinovac","Coronavac","inactivate"]

    for f in os.listdir(source_path):
        if not f.endswith('.csv'):
            continue
        domain = os.path.abspath(source_path)
        csv_file = os.path.join(domain, f)
        filename = os.path.basename(csv_file).split('.')[0]
        data = pd.read_csv(csv_file, dtype=str, error_bad_lines=False)
        print('Processing '+csv_file)
        out_dest = out_path + '\\' + filename + '.csv'
        # if not os.path.exists(out_dest):
        #     os.mkdir(out_dest)
        findKeyword(data, keywords, out_dest, 1)




