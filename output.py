from __future__ import unicode_literals
import re
import unicodedata
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer

"""
#GoogleColabを使う場合
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/blindX
#使用ライブラリ
!pip install -qU torch==2.0.* torchtext==0.15.* torchvision==0.15.* torchaudio==2.0.* torchmetrics==0.11.* torchdata==0.6.* \
    transformers==4.26.1 pytorch_lightning==1.9.3 sentencepiece==0.1.97
"""

# GPU利用有無
USE_GPU = torch.cuda.is_available()
# 事前学習済みモデル
PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"
# 転移学習済みモデル
MODEL_DIR = "./model"
# トークナイザー（SentencePiece）
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)
# 学習済みモデル
trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
if USE_GPU:
    trained_model.cuda()

def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    #text = normalize_neologd(text)
    text = text.lower()
    return text

def read_title_body(file):
    next(file)

    next(file)
    title = next(file).decode("utf-8").strip()
    #title = normalize_text(remove_brackets(title))
    body = normalize_text(" ".join([line.decode("utf-8").strip() for line in file.readlines()]))
    return title, body


# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 各種ハイパーパラメータ
args_dict = dict(
    data_dir="./data",  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_NAME,
    tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    max_grad_norm=1.0,
    seed=42,
)

# 学習に用いるハイパーパラメータを設定する
args_dict.update({
    "max_input_length":  512,  # 入力文の最大トークン数
    "max_target_length": 64,  # 出力文の最大トークン数
    "train_batch_size":  8,  # 訓練時のバッチサイズ
    "eval_batch_size":   8,  # テスト時のバッチサイズ
    "num_train_epochs":  1,  # 訓練するエポック数
    })
args = argparse.Namespace(**args_dict)

while True:
  body = input("入力.")
  if(body=="おわり"):#「おわり」入力でループ脱出
    break

  MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
  MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数

  def preprocess_body(text):
      return normalize_text(text.replace("\n", " "))

  # 推論モード設定
  trained_model.eval()

  # 前処理とトークナイズを行う
  inputs = [preprocess_body(body)]
  batch = tokenizer.batch_encode_plus(
      inputs, max_length=MAX_SOURCE_LENGTH, truncation=True,
      padding="longest", return_tensors="pt")

  input_ids = batch['input_ids']
  input_mask = batch['attention_mask']
  if USE_GPU:
      input_ids = input_ids.cuda()
      input_mask = input_mask.cuda()

  # 生成処理を行う
  outputs = trained_model.generate(
      input_ids=input_ids, attention_mask=input_mask,
      max_length=MAX_TARGET_LENGTH,
      temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
      num_beams=10,             # ビームサーチの探索幅
      diversity_penalty=3.0,    # 生成結果の多様性を生み出すためのペナルティ
      num_beam_groups=10,       # ビームサーチのグループ数
      num_return_sequences=1,  # 生成する文の数
      repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
  )

  # 生成されたトークン列を文字列に変換する
  generated_titles = [tokenizer.decode(ids, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
                      for ids in outputs]

  # 生成されたタイトルを表示する
  for i, title in enumerate(generated_titles):
      print(f"出力. {title}")
