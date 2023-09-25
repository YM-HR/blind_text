import re
import mojimoji
import csv

import MeCab
tagger = MeCab.Tagger("-Owakati")

import jaconv

def hiragana(total_text):
    mecab = MeCab.Tagger("-Ochasen")
    mecab.parse('') # 空でパースする必要がある
    node = mecab.parseToNode(total_text)
    text_box2 = []
    greek = [chr(i) for i in range(913, 969)]
    kazu = ['零','一','二','三','四','五','六','七','八','九','十','百','千','万','億','兆']
    
    mecab_greek = MeCab.Tagger()
    mecab_greek.parse('')

    while node :
        origin = node.surface #元の単語を代入
        #print(node.feature)
        try:
            if (origin in greek) and (node.feature.split(",")[0] in ["名詞"]):
                try:
                    node_greek = mecab_greek.parse(origin).split()
                    yomi = node_greek[1].split(",")[7]
                    text_box2.append(yomi)
                except:
                    yomi = input(origin + "←読み仮名をひらがなで入力：")
                    text_box2.append(yomi)
            elif ((node.feature.split(",")[1] in ["数"])and not(origin in kazu)) or ((node.feature.split(",")[1] in ["アルファベット"]) and not(origin in greek)):
                yomi = node.feature.split(",")[6] # 読み仮名を代入
                text_box2.append(yomi)             
            else:
                yomi = node.feature.split(",")[7] # 読み仮名を代入
                text_box2.append(yomi)
        except:
            yomi = origin
            text_box2.append(yomi)

        node = node.next

    #word = jaconv.kata2hira(''.join(text_box[1:-1]))
    word_h = jaconv.kata2hira(''.join(text_box2[1:-1]))
    word_ht = jaconv.kata2hira(' '.join(text_box2[1:-1]))
    text = word_ht
    if '_ TAB _' in word_ht:
        # 囲まれている部分を抽出
        #r = re.findall(r'_TAB_(.+)_TAB_', word_h)  # パターンに当てはまるものを全て抽出
        r = re.findall(r'_TAB_+(.+?)_TAB_',  word_h)
        x = re.findall(r'_ TAB _.+?_ TAB _', word_ht)
        for i in range(len(x)):
            word_ht = re.sub(x[i], r[i], word_ht)
    else: pass
    word_h = re.sub('_TAB_', '',word_h)

    if '_ TAB _' in word_ht:# 誤差確認用
        #print(r,x)
        #print(text)
        #print(word_ht)
        word_ht = re.sub('_ TAB _ ','',word_ht)
        #print(word_ht)
    
    return word_h,word_ht

def fukidashi_seikei(text):
    if ((text[0]=="「") and not(text[-1]== "」")):text = text[1:]
    elif (not(text[0]=="「") and (text[-1]== "」")):text = text[:-1]
    if (text.count('「') > text.count('」')) :text = text+"」"
    elif (text.count('「') < text.count('」')) :text = "「"+text
    text = "「"+ text +"」"
    if ((text.count('「') == 1)and(text.count('」') == 1))or('「「' in text) :text = text[1:-1]
    return text    

def re_text(shoki_text):
    #2.テキストから漢字交じりとひらがな(全角文字)の２つのデータを作成
    for i in ['kanji','hiragana']:
        main_text = str(shoki_text)
        main_text = re.sub(r'\ufeff', '', main_text)

        if i=='kanji':
            main_text = mojimoji.han_to_zen(mojimoji.zen_to_han(main_text, kana=False), digit=False, ascii=False)
            lines_kanji = [x.strip() for x in main_text.splitlines()]
            lines_kanji_tab = [x.strip() for x in main_text.splitlines()]

            list_num = 0
            x = 0
            for text in lines_kanji:
                text = fukidashi_seikei(text)
                text_kanji = re.sub('_TAB_', '',text)
                lines_kanji[list_num] = text_kanji

                text_kanji_tab = tagger.parse(text)#str型で、単語が空白で別れる
                if '_TAB_' in text:
                    # 囲まれている部分を抽出
                    r = re.findall(r'_TAB_(.+?)_TAB_', text)  # パターンに当てはまるものを全て抽出
                    #x = re.search(r'_ TAB _.+?_ TAB _',text_kanji_tab).group(0)
                    test = text_kanji_tab
                    x = re.findall(r'_ TAB _.+?_ TAB _', text_kanji_tab)
                    for i in range(len(x)):
                        text_kanji_tab = re.sub(x[i], r[i], text_kanji_tab)
                    #text_kanji_tab = re.sub(x, str(r)[2:-2], text_kanji_tab)
                else: pass
                if '_ TAB _' in text_kanji_tab:# 誤差確認用
                    #print(r,x)
                    #print(test)
                    pass
                text_kanji_tab = re.sub("\n", "", text_kanji_tab)
                lines_kanji_tab[list_num] = text_kanji_tab

                list_num+=1
        elif i=='hiragana':
            main_text = mojimoji.han_to_zen(main_text)
            main_text = re.sub('＿ＴＡＢ＿','_TAB_',main_text)
            lines = [x.strip() for x in main_text.splitlines()]
            lines_hiragana = []
            lines_hiragana_tab = []
            list_num = 0
            for text in lines:
                text = fukidashi_seikei(text)
                lines[list_num] = text
                list_num+=1
            for text_l in lines:
                text_h,text_ht = hiragana(text_l)

                lines_hiragana.append(text_h)
                lines_hiragana_tab.append(text_ht)
    return lines_kanji,lines_hiragana,lines_kanji_tab,lines_hiragana_tab


#ファイルの読み込み
path =  "./text.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

#例
lines = ["明日の天気は晴れのち曇り","隣の客はよく柿食う客だ","吾輩は猫である。"]

for text in lines[0:]:
    lines_kanji,lines_hiragana,lines_kanji_tab,lines_hiragana_tab  = re_text(text)
    csv_text= []
    #漢字交じりとひらがな(全角文字)をリストで対にそろえる
    for j in zip(lines_kanji,lines_hiragana,lines_kanji_tab,lines_hiragana_tab):
        csv_text.append(j)

        csv_path1 = './sample.csv'##5,045,101文 a
        for csv_path in[csv_path1]:
            with open(csv_path, 'a' ,encoding="utf_8_sig", newline='') as file:
                #mywriter = csv.writer(file, delimiter=',')
                mywriter = csv.writer(file, delimiter='\t')
                mywriter.writerows(csv_text)
""""""
#列がそろっているか確認(もし、そろってない場合は上記の書き込みと下記の読み込みの「,」の部分を「\t」に変更)
with open("./sample.csv", "r", encoding="utf-8") as f:
            for line in f:
                #line = line.strip().split(",")
                line = line.strip().split("\t")
                try:
                    assert len(line) == 4
                    assert len(line[0]) > 0
                    assert len(line[1]) > 0
                    assert len(line[2]) > 0
                    assert len(line[3]) > 0
                except:
                    print("エラー部分：",line)