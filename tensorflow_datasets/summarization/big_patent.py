# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BIGPATENT Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re

import nltk
import tensorflow_datasets.public_api as tfds

_ENGLISH_WORDS = frozenset(nltk.corpus.words.words())

_CITATION = """
@misc{sharma2019bigpatent,
    title={BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization},
    author={Eva Sharma and Chen Li and Lu Wang},
    year={2019},
    eprint={1906.03741},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
BIGPATENT, consisting of 1.3 million records of U.S. patent documents
along with human written abstractive summaries.
Each US patent application is filed under a Cooperative Patent Classification
(CPC) code. There are nine such classification categories:
A (Human Necessities), B (Performing Operations; Transporting),
C (Chemistry; Metallurgy), D (Textiles; Paper), E (Fixed Constructions),
F (Mechanical Engineering; Lightning; Heating; Weapons; Blasting),
G (Physics), H (Electricity), and
Y (General tagging of new or cross-sectional technology)

There are two features:
  - description: detailed description of patent.
  - summary: Patent abastract.

"""

_URL = "https://drive.google.com/uc?export=download&id=1mwH7eSh1kNci31xduR4Da_XcmTE8B8C3"

_DOCUMENT = "description"
_SUMMARY = "abstract"

_CPC_DESCRIPTION = {
    "a": "Human Necessities",
    "b": "Performing Operations; Transporting",
    "c": "Chemistry; Metallurgy",
    "d": "Textiles; Paper",
    "e": "Fixed Constructions",
    "f": "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
    "g": "Physics",
    "h": "Electricity",
    "y": "General tagging of new or cross-sectional technology"
}


class BigPatentConfig(tfds.core.BuilderConfig):
  """BuilderConfig for BigPatent."""

  @tfds.core.disallow_positional_args
  def __init__(self, cpc_codes=None, **kwargs):
    """BuilderConfig for Wikihow.

    Args:
      cpc_codes: str, cpc_codes
      **kwargs: keyword arguments forwarded to super.
    """
    super(BigPatentConfig, self).__init__(
        # 1.0.0 lower cased tokenized words.
        # 2.0.0 cased raw strings.
        version=tfds.core.Version("2.0.0", "Updated to cased raw strings."),
        supported_versions=[tfds.core.Version("1.0.0")],
        **kwargs)
    self.cpc_codes = cpc_codes


class BigPatent(tfds.core.BeamBasedBuilder):
  """BigPatent datasets."""

  BUILDER_CONFIGS = [
      BigPatentConfig(
          cpc_codes="*",
          name="all",
          description="Patents under all categories."),
  ] + [
      BigPatentConfig(  # pylint:disable=g-complex-comprehension
          cpc_codes=k,
          name=k,
          description=("Patents under Cooperative Patent Classification (CPC)"
                       "{0}: {1}".format(k, v)),
      ) for k, v in sorted(_CPC_DESCRIPTION.items())
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            _DOCUMENT: tfds.features.Text(),
            _SUMMARY: tfds.features.Text()
        }),
        supervised_keys=(_DOCUMENT, _SUMMARY),
        homepage="https://evasharma.github.io/bigpatent/",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    dl_path = dl_manager.download_and_extract(_URL)
    split_types = ["train", "val", "test"]
    extract_paths = dl_manager.extract({
        k: os.path.join(dl_path, "bigPatentDataNonTokenized", k + ".tar.gz")
        for k in split_types
    })
    extract_paths = {k: os.path.join(extract_paths[k], k) for k in split_types}

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"path": extract_paths["train"]},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"path": extract_paths["val"]},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={"path": extract_paths["test"]},
        ),
    ]

  def _build_pcollection(self, pipeline, path=None):
    """Build PCollection of examples."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(row):
      json_obj = json.loads(row)
      yield json_obj["publication_number"], {
          _DOCUMENT: clean_description(json_obj[_DOCUMENT]),
          _SUMMARY: clean_abstract(json_obj[_SUMMARY])
      }

    file_pattern = os.path.join(path, self.builder_config.cpc_codes, "*")
    return (pipeline
            | "ReadTextIO" >> beam.io.textio.ReadFromText(file_pattern)
            | beam.FlatMap(_process_example))


# The preprocessing functions below are kindly provided by the authors of
#   BigPatent Dataset. They are modified in a few ways:
#    1) minor code formating changes.
#    2) enchant is replaced with nltk to detect english words.
#    3) remove excessive white space.

# Regex for cleaning the abstract and description fields of unwanted text
# spans.

fig_exp1 = re.compile(r"(FIG.)\s+(\d)(,*)\s*(\d*)")
fig_exp2 = re.compile(r"(FIGS.)\s+(\d)(,*)\s*(\d*)")
fig_exp3 = re.compile(r"(FIGURE)\s+(\d)(,*)\s*(\d*)")

line_num_exp = re.compile(r"\[(\d+)\]")
non_empty_lines = re.compile(r"^\s*\[(\d+)\]")
table_header = re.compile(r"^(\s*)TABLE\s+\d+(\s+(.*))?$")


def remove_excessive_whitespace(text):
  return " ".join([w for w in text.split(" ") if w])


def clean_abstract(text):
  """Cleans the abstract text."""
  text = re.sub(r"[\(\{\[].*?[\}\)\]]", "", text).strip()
  text = remove_excessive_whitespace(text)
  return text


def remove_referenecs(text):
  """Remove references from description text."""
  text = fig_exp1.sub(r"FIG\2 ", text)
  text = fig_exp2.sub(r"FIG\2 ", text)
  text = fig_exp3.sub(r"FIG\2 ", text)
  return text


def get_list_of_non_empty_lines(text):
  """Remove non-empty lines."""
  # Split into lines
  # Remove empty lines
  # Remove line numbers
  return [
      non_empty_lines.sub("", s).strip()
      for s in text.strip().splitlines(True)
      if s.strip()
  ]


def remove_tables(sentences):
  """Remove Tables from description text."""
  # Remove tables from text
  new_sentences = []
  i = 0
  table_start = 0
  # A table header will be a line starting with "TABLE" after zero or more
  # whitespaces, followed by an integer.
  # After the integer, the line ends, or is followed by whitespace and
  # description.
  while i < len(sentences):
    sentence = sentences[i]
    if table_start == 0:
      # Not inside a table
      # Check if it's start of a table
      if table_header.match(sentence):
        table_start = 1
      else:
        new_sentences.append(sentence)

    elif table_start == 1:
      words = sentence.strip("\t").split(" ")
      num_eng = 0
      for w in words:
        if not w.isalpha():
          continue
        if w in _ENGLISH_WORDS:
          num_eng += 1
          if num_eng > 20:
            # Table end condition
            table_start = 0
            new_sentences.append(sentence)
            break
    i += 1
  return new_sentences


def remove_lines_with_less_words(sentences):
  """Remove sentences with less than 10 words."""
  new_sentences = []
  for sentence in sentences:
    words = set(sentence.split(" "))
    if len(words) > 10:
      new_sentences.append(sentence)
  return new_sentences


def clean_description(text):
  """Clean the description text."""
  # split the text by newlines, keep only non-empty lines
  sentences = get_list_of_non_empty_lines(text)
  # remove tables from the description text
  sentences = remove_tables(sentences)
  # remove sentences with less than 10 words
  sentences = remove_lines_with_less_words(sentences)
  text = "\n".join(sentences)
  # remove references like FIG. 8, FIGS. 8, 8, FIG. 8-d
  text = remove_referenecs(text)
  # remove excessive whitespace
  text = remove_excessive_whitespace(text)
  return text
