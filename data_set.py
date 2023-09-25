#３つのデータセットを作成します
from pprint import pprint
import csv
import numpy as np
OVER_SIZE_LIMIT = 200_000_000
csv.field_size_limit(OVER_SIZE_LIMIT)
import re

# 以下、書きたい処理
def data_set():
    with open("./sample.csv", "r", encoding="utf_8_sig") as f:
        reader = csv.reader(f,delimiter = "\t")
        row = [r for r in reader]

    next_row = []
    for i in row:
        if (len(i) == 4)and(len(i[0]) > 0)and(len(i[1]) > 0)and(len(i[2]) > 0)and(len(i[3]) > 0):
            new_i = i
            count = 0
            for text in i:
                text = re.sub(r'\n', '', text)
                while "\t" in text:
                    text = re.sub(r'\t', '', text)
                new_i[count] = text
                count+=1
            next_row.append(new_i)
        else:
            print(i)

    # 1つ目と2つ目の列を抽出して新しいリストを作成
    next_row = [[i[0], i[1]] for i in next_row]

    new_row = []
    for i in next_row:
        if (len(i) == 2)and(len(i[0]) > 0)and(len(i[1]) > 0):
            new_i = i
            count = 0
            for text in i:
                while "\t" in text:
                    text = re.sub(r'\t', '', text)
                new_i[count] = text
                count+=1
            new_row.append(new_i)
        else:
            print(i)

    new_row = [[i[0], i[1]] for i in new_row]
    #new_row = [[i[0], i[1], 0] for i in new_row]
    print(len(new_row))
    np.random.shuffle(new_row)
    if len(new_row) > 1000000:
        data_size = len(new_row[:100000])
    elif len(new_row) < 1000000:
        data_size = len(new_row[:100])

    x = int(data_size*0.05)
    y = int(data_size*0.1)
    z = int(data_size)

    dev_l = new_row[0:x]
    test_l = new_row[x:y]
    train_l = new_row[y:z]

    with open('./data/dev.tsv', 'w', encoding="utf_8_sig",newline="") as f:
        writer = csv.writer(f,delimiter = "\t")
        writer.writerows(dev_l)

    with open('./data/train.tsv', 'w', encoding="utf_8_sig",newline="") as f:
        writer = csv.writer(f,delimiter = "\t")
        writer.writerows(train_l)

    with open('./data/test.tsv', 'w', encoding="utf_8_sig",newline="") as f:
        writer = csv.writer(f,delimiter = "\t")
        writer.writerows(test_l)

data_set()
