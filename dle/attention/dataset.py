import logging
import os
import re
import unicodedata
from collections import deque
from typing import Union, Optional, Tuple, Sequence

import gin
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

logger = logging.getLogger(__name__)


@gin.configurable
def create_dataset(data_path: Union[str, os.PathLike],
                   num_examples: Optional[int] = None,
                   train_batch_size: int = 64,
                   test_batch_size: int = 256,
                   test_size: float = 0.2,
                   ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Tokenizer, Tokenizer]:
    logger.info(f'Create dataset with:\n'
                f'\ttrain batch size: {train_batch_size}\n'
                f'\ttest_batch_size: {test_batch_size}\n'
                f'\ttest_size: {test_size}\n'
                f'\t{"Only for the first " + str(num_examples) + " sentences" if num_examples is not None else ""}')
    targ_lang, inp_lang = _load_data(data_path, num_examples)

    inputs, inp_tokenizer = _tokenize(inp_lang)
    targets, targ_tokenizer = _tokenize(targ_lang)

    inp_train, inp_val, targ_train, targ_val = train_test_split(inputs, targets,
                                                                test_size=test_size)
    train_dataset = (tf.data.Dataset.from_tensor_slices((inp_train, targ_train))
                                    .shuffle(len(inp_train))
                                    .batch(train_batch_size))
    val_dataset = (tf.data.Dataset.from_tensor_slices((inp_val, targ_val))
                                  .batch(test_batch_size))

    return train_dataset, val_dataset, inp_tokenizer, targ_tokenizer


def _load_data(data_path: Union[str, os.PathLike],
               num_examples: Optional[int] = None
               ) -> Tuple[Sequence[str], Sequence[str]]:
    lang1 = deque()
    lang2 = deque()
    with open(data_path, encoding='UTF-8') as data_file:
        for i, line in enumerate(data_file):
            if num_examples is not None and i >= num_examples:
                break

            s1, s2, _ = line.split('\t')
            lang1.append(_preprocess_sentence(s1))
            lang2.append(_preprocess_sentence(s2))

    logger.info(f'Loaded {len(lang1)} sentences for each language.')
    return lang1, lang2


def _preprocess_sentence(s: str) -> str:
    s = _unicode_to_ascii(s.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: 'he is a boy.' => 'he is a boy .'
    s = re.sub(r'([?.!,¿])', r' \1 ', s)
    s = re.sub(r'[" "]+', ' ', s)

    # replacing everything with space except
    # (a-z, A-Z, ".", "?", "!", ",", "'" and polish characters)
    s = re.sub(r'[^a-zA-Z?.!,¿łęąćżźóńś\']+', ' ', s)
    return '<start> ' + s.strip() + ' <end>'


def _unicode_to_ascii(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def _tokenize(lang: Sequence[str]) -> Tuple[tf.Tensor, Tokenizer]:
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(lang)

    lang = tokenizer.texts_to_sequences(lang)
    lang = tf.keras.preprocessing.sequence.pad_sequences(lang, padding='post')
    return lang, tokenizer
