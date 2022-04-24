from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter

@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]

@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray

@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    sent = []
    allign = []
    text = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            lin = line.replace('&', '&#38;')
            text.append(lin)
    xl = ' '.join(text)
    root = ET.fromstring(xl)
    for child in root:
        s = SentencePair(child[0].text.split(' '),child[1].text.split(' '))
        sent.append(s)
        sure_al = []
        poss_al = []
        if child[2].text != None:
            st = child[2].text.split(' ')
            for i in st:
                n = i.find('-')
                sure_al.append((int(i[:n]),int(i[n+1:])))
        if child[3].text != None:
            pt = child[3].text.split(' ')
            for j in pt:
                n = j.find('-')
                poss_al.append((int(j[:n]),int(j[n+1:])))
        l = LabeledAlignment(sure_al,poss_al)
        allign.append(l)
    return (sent, allign)

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_dict = {}
    target_dict = {}
    source_list = []
    target_list = []
    for sentence in sentence_pairs:
        for token_source in sentence.source:
            source_list.append(token_source)
        for token_target in sentence.target:
            target_list.append(token_target)
    source_dict, target_dict = Counter(source_list), Counter(target_list)
    source_dict = sorted(source_dict.items(), key = lambda x: x[1] , reverse = True)
    target_dict = sorted(target_dict.items(), key = lambda x: x[1] , reverse = True)
    if freq_cutoff != None:
        source_dict = source_dict[0:freq_cutoff]
        target_dict = target_dict[0:freq_cutoff]
    if freq_cutoff != 0:  
        source_dict = dict(zip(np.array(source_dict)[:,0],np.arange(len(source_dict))))
        target_dict = dict(zip(np.array(target_dict)[:,0],np.arange(len(target_dict))))
    else:
        source_dict = {}
        target_dict = {}
    return (source_dict, target_dict)


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    fuck = []
    for sentence in sentence_pairs:
        source = np.array([])
        target = np.array([])
        for token_source in sentence.source:
            if token_source in source_dict.keys():
                source = np.append(source, source_dict['{}'.format(token_source)])
        for token_target in sentence.target:
            if token_target in target_dict.keys():
                target = np.append(target, target_dict['{}'.format(token_target)])
        if (len(source) and len(target)) == 0:
            return []
        fuck.append(TokenizedSentencePair(source.astype(np.int32), target.astype(np.int32)))
    return fuck
