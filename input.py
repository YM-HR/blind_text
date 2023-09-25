from __future__ import unicode_literals
import re
import unicodedata
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm.auto import tqdm

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

"""
#GoogleColabを使う場合
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/blindX
#使用ライブラリ
!pip install -qU torch==2.0.* torchtext==0.15.* torchvision==0.15.* torchaudio==2.0.* torchmetrics==0.11.* torchdata==0.6.* \
    transformers==4.26.1 pytorch_lightning==1.9.3 sentencepiece==0.1.97
"""

import data_set
#dev,train.testデータセットを作成↓
#data_set.data_set()

# GPU利用有無
USE_GPU = torch.cuda.is_available()
# 事前学習済みモデル
PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"
# 転移学習済みモデル
MODEL_DIR = "./model"


def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)


    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def remove_brackets(text):
    text = re.sub(r"(^【[^】]*】)|(【[^】]*】$)", "", text)
    return text

def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

def read_title_body(file):
    next(file)

    next(file)
    title = next(file).decode("utf-8").strip()
    title = normalize_text(remove_brackets(title))
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


class TsvDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512):
        self.file_path = os.path.join(data_dir, type_path)

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, title, body):
        # タスク用の入出力形式に変換する。
        input = f"{body}"
        target = f"{title}"
        return input, target

    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")

                assert len(line) == 2
                assert len(line[0]) > 0
                assert len(line[1]) > 0

                title = line[0]
                body = line[1]

                input, target = self._make_record(title, body)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True,
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True,
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)


# トークナイザー（SentencePiece）モデルの読み込み
tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)
# テストデータセットの読み込み
train_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "train.tsv",input_max_len=512, target_max_len=64)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # 事前学習済みモデルの読み込み
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

        # トークナイザーの読み込み
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path, is_fast=True)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, type_path, args):
        """データセットを作成する"""
        return TsvDataset(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            type_path=type_path,
            input_max_len=args.max_input_length,
            target_max_len=args.max_target_length)

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer,
                                             type_path="train.tsv", args=self.hparams)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer,
                                           type_path="dev.tsv", args=self.hparams)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.train_batch_size,
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.eval_batch_size,
                          num_workers=4)


# 学習に用いるハイパーパラメータを設定する
args_dict.update({
    "max_input_length":  512,  # 入力文の最大トークン数
    "max_target_length": 64,  # 出力文の最大トークン数
    "train_batch_size":  8,  # 訓練時のバッチサイズ
    "eval_batch_size":   8,  # テスト時のバッチサイズ
    "num_train_epochs":  3,  # 訓練するエポック数
    })
args = argparse.Namespace(**args_dict)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_level="O1",
    amp_backend="apex",
    gradient_clip_val=args.max_grad_norm,
)

# 転移学習の実行
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

# 最終エポックのモデルを保存
model.tokenizer.save_pretrained(MODEL_DIR)
model.model.save_pretrained(MODEL_DIR)

del model


"""
# トークナイザー（SentencePiece）
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)
# 学習済みモデル
trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()


# テストデータの読み込み
test_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv",
                          input_max_len=args.max_input_length,
                          target_max_len=args.max_target_length)

test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

trained_model.eval()

inputs = []
outputs = []
targets = []

for batch in tqdm(test_loader):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    output = trained_model.generate(input_ids=input_ids,
        attention_mask=input_mask,
        max_length=args.max_target_length,
        temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
        repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

    output_text = [tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
                for ids in output]
    target_text = [tokenizer.decode(ids, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
                for ids in batch["target_ids"]]
    input_text = [tokenizer.decode(ids, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
                for ids in input_ids]

    inputs.extend(input_text)
    outputs.extend(output_text)
    targets.extend(target_text)


body = "わがはいはねこである。なまえはまだない。"

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
    temperature=2.0,          # 生成にランダム性を入れる温度パラメータ
    num_beams=10,             # ビームサーチの探索幅
    diversity_penalty=3.0,    # 生成結果の多様性を生み出すためのペナルティ
    num_beam_groups=10,       # ビームサーチのグループ数
    num_return_sequences=10,  # 生成する文の数
    repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
)

# 生成されたトークン列を文字列に変換する
generated_titles = [tokenizer.decode(ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
                    for ids in outputs]

# 生成されたタイトルを表示する
for i, title in enumerate(generated_titles):
    print(f"{i+1:2}. {title}")
"""