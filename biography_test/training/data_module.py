from typing import List, Tuple, Dict, Any
import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import transformers
from training.utility import DataArguments, AdditionalTrainingArguments, attribute_list, set_seed
from training.utility import QARawData, PreTrainingPersonIndexInfoList, FineTuningTrainPersonIndexInfoList
from training.utility import AttentionMaskType
from training.utility import FirstTokenAccuracyCalculationStrategy
from training.utility import first_token_accuracy_calculation_interval_when_only_end
from training.utility import construct_selected_person_index_set
from transformers.trainer_pt_utils import LabelSmoother
from tqdm import trange, tqdm
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
import random

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class BiographyDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: Dict[str, str],
                 additional_training_args: AdditionalTrainingArguments):
        self.tokenizer = tokenizer
        self.additional_training_args = additional_training_args
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        self.biography_index_to_token_ids = {}
        # the code is written under the assumption that the tokenizer is GPT2Tokenizer
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        # concatenate the biography data
        for biography_index, biography in tqdm(raw_data.items(), desc='Biography data tokenizing'):
            assert biography[0] == ' ', ('Each biography should start with a space, so that the tokenizer result is '
                                         'correct.')
            self.biography_index_to_token_ids[biography_index] = tokenizer(biography + tokenizer.eos_token)['input_ids']
        # this dataset will not be used in the training process, it is set to pass to validation of huggingface Trainer
        self.construct_dataset()

    def construct_dataset(self) -> None:
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        biography_key_list = list(self.biography_index_to_token_ids.keys())
        random.shuffle(biography_key_list)
        biography_ids = []
        for biography_key in biography_key_list:
            biography_ids.extend(self.biography_index_to_token_ids[biography_key])
        # attention_mask 0: 2D attention mask for debugging
        # attention_mask = [[True] * tokenizer.model_max_length for _ in range(tokenizer.model_max_length)]
        # attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        for i in trange(0, len(biography_ids), self.tokenizer.model_max_length, desc='Biography data chunking'):
            ids_chunk = biography_ids[i:i + self.tokenizer.model_max_length]
            if len(ids_chunk) == self.tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
                target = input_ids.clone()
            else:
                input_ids = torch.tensor(
                    ids_chunk + [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
                target = torch.tensor(
                    ids_chunk + [IGNORE_TOKEN_ID] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)

            # https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/models/gpt2/modeling_gpt2.py#L1446
            target_mask = torch.cat(
                (torch.tensor([True], dtype=torch.bool), input_ids.ne(self.tokenizer.pad_token_id)), dim=0)[:-1]
            target = target.masked_fill(~target_mask, IGNORE_TOKEN_ID)

            # region construct attention mask
            match self.additional_training_args.attention_mask_type:
                case AttentionMaskType.MASK_EOS:
                    # attention_mask 1: only mask the eos token
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                case AttentionMaskType.ALL_TRUE:
                    # attention_mask 2: no token will be masked
                    attention_mask = torch.tensor([True] * len(input_ids), dtype=torch.bool)
                case _:
                    raise ValueError('Invalid attention mask type.')
                    # attention_mask 3: mask previous biography entries
                    # https://github.com/pytorch/pytorch/issues/110213
                    # ensure that there would not be a row of all False in the attention mask
                    # eos token is not masked in the attention mask. however, it is masked in the label
                    # solution of huggingface:
                    # https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/modeling_attn_mask_utils.py#L402
                    # current_attention_mask_prefix = []
                    # attention_mask_of_each_token_list = []
                    # for token_id in input_ids:
                    #     attention_mask_of_each_token_list.append(
                    #         current_attention_mask_prefix +  # previous token
                    #         [True] +  # current token
                    #         [False] * (tokenizer.model_max_length - len(current_attention_mask_prefix) - 1)
                    #     )  # padding
                    #     if token_id != tokenizer.pad_token_id:
                    #         current_attention_mask_prefix.append(True)
                    #     else:
                    #         current_attention_mask_prefix = [False] * (len(current_attention_mask_prefix) + 1)
                    # attention_mask = torch.tensor(attention_mask_of_each_token_list, dtype=torch.bool)
            # endregion

            self.input_ids_list.append(input_ids)
            self.label_list.append(target)
            self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }


class PreTrainingFirstTokenAccuracyDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: Dict,
                 additional_training_args: AdditionalTrainingArguments):
        self.input_ids_list, self.token_position_list, self.attention_mask_list = [], [], []
        # assert isinstance(tokenizer, GPTNeoXTokenizerFast)
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        biography_ids = []
        attribute_first_token_position = []
        raw_data_list = list(raw_data.values())
        random.shuffle(raw_data_list)
        for info_dict in tqdm(raw_data_list, desc='Biography data tokenizing (first token)'):
            biography = info_dict['biography'] + tokenizer.eos_token
            token_info = info_dict['token_info']
            single_person_biography_ids = tokenizer(biography)['input_ids']
            single_person_attribute_first_token_position: List[None | Tuple[str, int]] = (
                    [None] * len(single_person_biography_ids))
            for attribute, first_token_position_info in token_info.items():
                single_person_attribute_first_token_position[first_token_position_info['first_token_position']] = \
                    (attribute, first_token_position_info['first_token'])
            biography_ids.extend(single_person_biography_ids)
            attribute_first_token_position.extend(single_person_attribute_first_token_position)
        for i in trange(0, len(biography_ids), tokenizer.model_max_length, desc='Biography data chunking (first token)'):
            ids_chunk = biography_ids[i:i + tokenizer.model_max_length]
            first_token_position_chunk = attribute_first_token_position[i:i + tokenizer.model_max_length]
            if len(ids_chunk) == tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
            else:
                input_ids = torch.tensor(
                    ids_chunk + [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            token_position = {}
            for attribute in attribute_list:
                token_position[attribute] = {
                    'index': torch.tensor([False] * len(input_ids), dtype=torch.bool),
                    'first_token_list': []
                }
            for info_id in range(len(first_token_position_chunk)):
                info = first_token_position_chunk[info_id]
                if info is None:
                    continue
                attribute, first_token = info
                assert input_ids[info_id] == first_token
                token_position[attribute]['index'][info_id] = True
                token_position[attribute]['first_token_list'].append(first_token)
            # region construct attention mask
            match additional_training_args.attention_mask_type:
                case AttentionMaskType.ALL_TRUE:
                    attention_mask = torch.tensor([True] * len(input_ids), dtype=torch.bool)
                case _:
                    raise ValueError('Invalid attention mask type.')
            # endregion
            self.input_ids_list.append(input_ids)
            self.token_position_list.append(token_position)
            self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'token_position': self.token_position_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }


class QADataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: QARawData,
                 max_length: int):
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        for person_index in tqdm(raw_data, desc='Constructing QADataset'):
            # if len(self.input_ids_list) > 5000: break
            for attribute in raw_data[person_index]:
                prompt = raw_data[person_index][attribute]['prompt']
                answer = raw_data[person_index][attribute]['answer']
                assert prompt[-1] == ':'
                prompt_ids = tokenizer(prompt)['input_ids']
                answer_ids = tokenizer(answer)['input_ids']
                assert len(prompt_ids) + len(answer_ids) + 1 <= max_length  # +1 for <eos>
                assert len(prompt_ids) + len(answer_ids) == len(tokenizer(prompt + answer)['input_ids'])
                pad_length = max_length - (len(prompt_ids) + len(answer_ids) + 1)
                assert pad_length >= 0
                input_ids = torch.tensor(
                    prompt_ids + answer_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * pad_length,
                    dtype=torch.int64
                )
                target_ids = torch.tensor(
                    [IGNORE_TOKEN_ID] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id] +
                    [IGNORE_TOKEN_ID] * pad_length,
                    dtype=torch.int64
                )
                # although the token indicating <eos> will also be masked, it is not a problem
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                self.input_ids_list.append(input_ids)
                self.label_list.append(target_ids)
                self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

class QAPretrainingDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_qa_data: QARawData,
                 additional_training_args: AdditionalTrainingArguments):
        self.tokenizer = tokenizer
        self.additional_training_args = additional_training_args
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []  # 与BiographyDataset字段对齐
        self.qa_index_to_token_ids = {}  # 类似BiographyDataset的biography_index_to_token_ids
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        self.construct_qa_token_ids(raw_qa_data)
        self.construct_dataset()

    def construct_qa_token_ids(self, raw_qa_data: QARawData) -> None:
        for person_index in tqdm(raw_qa_data, desc='QA data tokenizing'):
            person_data = raw_qa_data[person_index]
            for attr in person_data:
                # 拼接prompt+answer+eos_token
                qa_text = person_data[attr]['prompt'] + person_data[attr]['answer'] + self.tokenizer.eos_token
                # 存储为{person_index_attr: token_ids}
                qa_key = f"{person_index}_{attr}"
                self.qa_index_to_token_ids[qa_key] = self.tokenizer(qa_text)['input_ids']

    def construct_dataset(self) -> None:
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        qa_key_list = list(self.qa_index_to_token_ids.keys())
        random.shuffle(qa_key_list)  # 打乱QA对顺序（与BiographyDataset的random.shuffle一致）
        qa_ids = []
        for qa_key in qa_key_list:
            qa_ids.extend(self.qa_index_to_token_ids[qa_key])  # 拼接所有QA的token ID（类似BiographyDataset的biography_ids拼接）
        
        # 按model_max_length分块（与BiographyDataset的分块逻辑一致）
        for i in trange(0, len(qa_ids), self.tokenizer.model_max_length, desc='QA data chunking'):
            ids_chunk = qa_ids[i:i + self.tokenizer.model_max_length]
            # 处理完整块与不完整块（与BiographyDataset的input_ids和target生成逻辑一致）
            if len(ids_chunk) == self.tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
                target = input_ids.clone()
            else:
                input_ids = torch.tensor(
                    ids_chunk + [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
                target = torch.tensor(
                    ids_chunk + [IGNORE_TOKEN_ID] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            
            # 标签掩码处理
            target_mask = torch.cat(
                (torch.tensor([True], dtype=torch.bool), input_ids.ne(self.tokenizer.pad_token_id)), dim=0)[:-1]
            target = target.masked_fill(~target_mask, IGNORE_TOKEN_ID)
            
            # 注意力掩码生成
            match self.additional_training_args.attention_mask_type:
                case AttentionMaskType.MASK_EOS:
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                case AttentionMaskType.ALL_TRUE:
                    attention_mask = torch.tensor([True] * len(input_ids), dtype=torch.bool)
                case _:
                    raise ValueError('Invalid attention mask type.')
            
            self.input_ids_list.append(input_ids)
            self.label_list.append(target)
            self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return { 
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

class QAFirstTokenAccuracyDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: QARawData,
                 max_length: int):
        self.input_ids_list, self.token_position_list, self.attention_mask_list = [], [], []
        # assert isinstance(tokenizer, GPTNeoXTokenizerFast)
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        for person_index in tqdm(raw_data, desc='Constructing QAFirstTokenAccuracyDataset'):
            # if len(self.input_ids_list) > 5000: break
            for attribute in raw_data[person_index]:
                # the validation of the dataset is completed by QADataset
                prompt_ids = tokenizer(raw_data[person_index][attribute]['prompt'])['input_ids']
                answer_ids = tokenizer(raw_data[person_index][attribute]['answer'])['input_ids']
                pad_length = max_length - (len(prompt_ids) + len(answer_ids) + 1)
                input_ids = torch.tensor(
                    prompt_ids + answer_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * pad_length,
                    dtype=torch.int64
                )
                token_position = {}
                for _attribute in attribute_list:
                    token_position[_attribute] = {
                        'index': torch.tensor([False] * len(input_ids), dtype=torch.bool),
                        'first_token_list': []
                    }
                token_position[attribute]['index'][len(prompt_ids)] = True
                token_position[attribute]['first_token_list'].append(answer_ids[0])
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                self.input_ids_list.append(input_ids)
                self.token_position_list.append(token_position)
                self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'token_position': self.token_position_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

# region original code of calculating exact match accuracy
# class QAExactMatchDataset(Dataset):
#     def __init__(self,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  raw_data: QARawData,
#                  max_length: int,
#                  attribute: str):
#         self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
#         assert isinstance(tokenizer, GPTNeoXTokenizerFast)
#         for person_index in tqdm(raw_data, desc=f'Constructing QA dataset of attribute {attribute}'):
#             # the validation of the dataset is completed by QADataset
#             prompt_ids = tokenizer(raw_data[person_index][attribute]['prompt'])['input_ids']
#             answer_ids = tokenizer(raw_data[person_index][attribute]['answer'])['input_ids']
#             pad_length = max_length - (len(prompt_ids) + len(answer_ids) + 1)
#             input_ids = torch.tensor(
#                 prompt_ids + answer_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * pad_length,
#                 dtype=torch.int64
#             )
#             label_ids = torch.tensor(
#                 prompt_ids[1:] + answer_ids + [tokenizer.pad_token_id] * (pad_length + 1),
#                 dtype=torch.int64
#             )
#             label_mask = torch.tensor(
#                 [True] * len(prompt_ids[1:]) + [False] * (len(answer_ids) + 1) + [True] * pad_length,
#             )
#             attention_mask = input_ids.ne(tokenizer.pad_token_id)
#             self.input_ids_list.append(input_ids)
#             self.label_list.append({
#                 'ids': label_ids,
#                 'mask': label_mask
#             })
#             self.attention_mask_list.append(attention_mask)
#
#     def __len__(self):
#         return len(self.input_ids_list)
#
#     def __getitem__(self, idx):
#         return {
#             'input_ids': self.input_ids_list[idx],
#             'labels': self.label_list[idx],
#             'attention_mask': self.attention_mask_list[idx],
#         }
#
#
# def construct_qa_exact_match_data_module(
#         tokenizer: transformers.PreTrainedTokenizer,
#         data_args: DataArguments,
#         max_length: int) -> Tuple[Dict[str, QAExactMatchDataset], Dict[str, QAExactMatchDataset]]:
#     train_raw_data: QARawData = json.load(open(data_args.train_qa_data_path))
#     test_raw_data: QARawData = json.load(open(data_args.test_qa_data_path))
#     train_dataset_dict, test_dataset_dict = {}, {}
#     for attribute in attribute_list:
#         train_dataset_dict[attribute] = QAExactMatchDataset(tokenizer, train_raw_data, max_length, attribute)
#         test_dataset_dict[attribute] = QAExactMatchDataset(tokenizer, test_raw_data, max_length, attribute)
#     return train_dataset_dict, test_dataset_dict

# endregion

class QAExactMatchDataset(Dataset):
    def __init__(self,
                 raw_data: QARawData, ):
        self.prompt_list, self.answer_list, self.attribute_list = [], [], []
        for person_index in tqdm(raw_data, desc='Constructing QAExactMatchDataset'):
            for attribute in raw_data[person_index]:
                # the tokenization of the dataset is left in the callback
                self.prompt_list.append(raw_data[person_index][attribute]['prompt'])
                self.answer_list.append(raw_data[person_index][attribute]['answer'])
                self.attribute_list.append(attribute)

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            'prompt': self.prompt_list[idx],
            'answer': self.answer_list[idx],
            'attribute': self.attribute_list[idx],
        }


def filter_biography_data_with_token_info(data_with_token_info: Dict[str, Any],
                                          person_index_info_list: PreTrainingPersonIndexInfoList) -> Dict[str, Any]:
    selected_person_index_set = construct_selected_person_index_set(person_index_info_list)
    result = {}
    for biography_index in data_with_token_info:
        person_index, _ = biography_index.split('_')
        if int(person_index) in selected_person_index_set:
            result[biography_index] = data_with_token_info[biography_index]
    return result


def construct_pre_training_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments) -> BiographyDataset:
    # load bio data
    biography_data: Dict = json.load(open(data_args.biography_data_path))
    biography_data: Dict = filter_biography_data_with_token_info(
        biography_data, additional_training_args.pre_training_person_index_info_list)
    raw_data: Dict[str, Any] = {}
    for biography_index in biography_data:
        assert tokenizer.__class__.__name__ == biography_data[biography_index]['tokenizer']
        raw_data[biography_index] = biography_data[biography_index]['biography']
    return BiographyDataset(tokenizer, raw_data, additional_training_args)


def construct_pre_training_first_token_accuracy_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments) -> PreTrainingFirstTokenAccuracyDataset:
    data_with_token_info = json.load(open(data_args.biography_data_path))
    data_with_token_info = filter_biography_data_with_token_info(
        data_with_token_info, additional_training_args.pre_training_person_index_info_list)
    assert tokenizer.__class__.__name__ == list(data_with_token_info.values())[0]['tokenizer']
    return PreTrainingFirstTokenAccuracyDataset(tokenizer, data_with_token_info, additional_training_args)


def filter_qa_data_with_token_info(raw_data: QARawData,
                                   person_index_info_list: FineTuningTrainPersonIndexInfoList) -> QARawData:
    selected_person_index_set = construct_selected_person_index_set(person_index_info_list)
    result = {}
    for person_index in raw_data:
        if int(person_index) in selected_person_index_set:
            result[person_index] = raw_data[person_index]
    return result


def construct_qa_fine_tuning_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments,
        max_length: int) -> QADataset:
    raw_data: QARawData = json.load(open(data_args.all_qa_data_path))
    raw_data = filter_qa_data_with_token_info(raw_data,
                                              additional_training_args.fine_tuning_training_person_index_info_list)
    return QADataset(tokenizer, raw_data, max_length)

def construct_qa_pre_training_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments) -> QAPretrainingDataset:
    raw_data: QARawData = json.load(open(data_args.all_qa_data_path))
    raw_data = filter_qa_data_with_token_info(raw_data,
                                              additional_training_args.pre_training_qa_info_list)
    return QAPretrainingDataset(tokenizer, raw_data, additional_training_args)

# I deleted this code
def construct_qa_first_token_accuracy_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments,
        max_length: int
) -> QAFirstTokenAccuracyDataset:
    raw_data: QARawData = json.load(open(data_args.all_qa_data_path))
    _raw_data = filter_qa_data_with_token_info(
        raw_data, additional_training_args.pre_training_qa_info_list)
    return QAFirstTokenAccuracyDataset(tokenizer, _raw_data, max_length)


def construct_qa_exact_match_data_module(
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments
) -> QAExactMatchDataset:
    raw_data: QARawData = json.load(open(data_args.all_qa_data_path))
    test_raw_data = filter_qa_data_with_token_info(raw_data, additional_training_args.test_person_index_info_list)
    return QAExactMatchDataset(test_raw_data)

# validation part
# def pre_training_dataset_validation():
#     torch.set_printoptions(profile="full")  # print complete tensor
#     tokenizer = AutoTokenizer.from_pretrained(
#         "./model/gpt-neox/pythia-70m",
#         model_max_length=512,
#         padding_side="right",
#         use_fast=True,
#     )
#     assert tokenizer.pad_token is None
#     tokenizer.pad_token = tokenizer.unk_token
#     data_args = DataArguments(
#         biography_data_path="./data/processed_0720_v0730/biography/single.json",
#         all_qa_data_path="./data/processed_0720_v0730/qa/all.json",
#     )
#     additional_training_args = AdditionalTrainingArguments(
#         attention_mask_type=AttentionMaskType.ALL_TRUE,
#         pre_training_person_index_info_list=[
#             {"start": 50, "end": 100, },
#             {"start": 100, "end": 150, },
#             {"start": 250, "end": 300, }
#         ]
#     )
#     biography_dataset = construct_pre_training_data_module(tokenizer, data_args, additional_training_args)
#     biography_dataset.construct_dataset()
#     print(biography_dataset[0])


# def pre_training_first_token_accuracy_dataset_validation():
#     set_seed()
#     torch.set_printoptions(profile="full")  # print complete tensor
#     tokenizer = AutoTokenizer.from_pretrained(
#         "./model/gpt-neox/pythia-70m",
#         model_max_length=512,
#         padding_side="right",
#         use_fast=True,
#     )
#     assert tokenizer.pad_token is None
#     tokenizer.pad_token = tokenizer.unk_token
#     data_args = DataArguments(
#         biography_data_path="./data/processed_0720_v0730/biography/single.json",
#         all_qa_data_path="./data/processed_0720_v0730/qa/all.json",
#     )
#     additional_training_args = AdditionalTrainingArguments(
#         attention_mask_type=AttentionMaskType.ALL_TRUE,
#         pre_training_person_index_info_list=[
#             {"start": 50, "end": 100, },
#             {"start": 100, "end": 150, },
#             {"start": 250, "end": 300, }
#         ]
#     )
#     first_token_accuracy_dataset = construct_pre_training_first_token_accuracy_data_module(
#         tokenizer, data_args, additional_training_args)
#     print(first_token_accuracy_dataset[0])


# def qa_fine_tuning_dataset_validation():
#     torch.set_printoptions(profile="full")  # print complete tensor
#     tokenizer = AutoTokenizer.from_pretrained(
#         "./model/gpt-neox/pythia-70m",
#         model_max_length=32,
#         padding_side="right",
#         use_fast=True,
#     )
#     tokenizer.pad_token = tokenizer.unk_token
#     data_args = DataArguments(
#         all_qa_data_path="./data/processed_0720_v0730/qa/all.json",
#     )
#     additional_training_args = AdditionalTrainingArguments(
#         fine_tuning_training_person_index_info_list=[
#             # {"start": 50, "end": 100, },
#             # {"start": 100, "end": 150, },
#             # {"start": 250, "end": 300, }
#             {"start": 0, "end": 200000, }
#         ],
#         fine_tuning_test_person_index_info_dict={
#             "task_0": [{"start": 100000, "end": 100050}],
#             "task_1": [{"start": 120000, "end": 120050}],
#         }
#     )
#     qa_dataset = construct_qa_fine_tuning_data_module(tokenizer, data_args, additional_training_args, 32)
#     print(qa_dataset[0])


# def qa_first_token_accuracy_dataset_validation():
#     torch.set_printoptions(profile="full")  # print complete tensor
#     tokenizer = AutoTokenizer.from_pretrained(
#         "./model/gpt-neox/pythia-70m",
#         model_max_length=32,
#         padding_side="right",
#         use_fast=True,
#     )
#     tokenizer.pad_token = tokenizer.unk_token
#     data_args = DataArguments(
#         all_qa_data_path="./data/processed_0720_v0730/qa/all.json",
#     )
#     additional_training_args = AdditionalTrainingArguments(
#         fine_tuning_training_person_index_info_list=[
#             {"start": 50, "end": 100, },
#             {"start": 100, "end": 150, },
#             {"start": 250, "end": 300, }
#         ],
#         fine_tuning_test_person_index_info_dict={
#             "task_0": [{"start": 100000, "end": 100050}],
#             "task_1": [{"start": 120000, "end": 120050}],
#         }
#     )
#     dataset_info_list = construct_qa_first_token_accuracy_data_module(
#         tokenizer, data_args, additional_training_args, 32)
#     print(dataset_info_list[0]['dataset'][0])


# def qa_exact_match_dataset_validation():
#     data_args = DataArguments(
#         all_qa_data_path="./data/processed_0720_v0730/qa/all.json",
#     )
#     additional_training_args = AdditionalTrainingArguments(
#         fine_tuning_training_person_index_info_list=[
#             {"start": 50, "end": 100, },
#             {"start": 100, "end": 150, },
#             {"start": 250, "end": 300, }
#         ],
#         fine_tuning_test_person_index_info_dict={
#             "task_0": [{"start": 100000, "end": 100050}],
#             "task_1": [{"start": 120000, "end": 120050}],
#         }
#     )
#     dataset_info_list = construct_qa_exact_match_data_module(data_args, additional_training_args)
#     print(dataset_info_list[0]['dataset'][0])

class MixedDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 bio_raw_data: Dict[str, str],
                 qa_raw_data: QARawData,
                 additional_training_args: AdditionalTrainingArguments,
                 qar: float):
        self.tokenizer = tokenizer
        self.additional_training_args = additional_training_args
        self.qar = qar
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        self.biography_index_to_token_ids = {}
        self.qa_index_to_token_ids = {}
        # the code is written under the assumption that the tokenizer is GPT2Tokenizer
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        # concatenate the biography data
        for biography_index, biography in tqdm(bio_raw_data.items(), desc='Biography data tokenizing'):
            assert biography[0] == ' ', ('Each biography should start with a space, so that the tokenizer result is '
                                         'correct.')
            self.biography_index_to_token_ids[biography_index] = tokenizer(biography + tokenizer.eos_token)['input_ids']
        # concatenate the qa data
        for person_index in tqdm(qa_raw_data, desc='QA data tokenizing'):
            for attr in qa_raw_data[person_index]:
                qa_text = qa_raw_data[person_index][attr]['prompt'] + qa_raw_data[person_index][attr]['answer'] + tokenizer.eos_token
                qa_key = f"{person_index}_{attr}"
                self.qa_index_to_token_ids[qa_key] = tokenizer(qa_text)['input_ids']

        self.construct_dataset()
    
    def construct_dataset(self) -> None:
        # shuffle 1
        shuffled_bio_keys = list(self.biography_index_to_token_ids.keys())
        random.shuffle(shuffled_bio_keys)
        shuffled_qa_keys = list(self.qa_index_to_token_ids.keys())
        random.shuffle(shuffled_qa_keys)

        biography_ids = []
        for biography_key in shuffled_bio_keys:
            biography_ids.extend(self.biography_index_to_token_ids[biography_key])
        qa_ids = []
        for qa_key in shuffled_qa_keys:
            qa_ids.extend(self.qa_index_to_token_ids[qa_key])

        bio_input_ids_list, bio_label_list, bio_attention_mask_list = self._process_dataset(biography_ids)
        qa_input_ids_list, qa_label_list, qa_attention_mask_list = self._process_dataset(qa_ids)
        assert len(bio_input_ids_list) == len(bio_label_list) and len(bio_label_list) == len(bio_attention_mask_list)
        assert len(qa_input_ids_list) == len(qa_label_list) and len(qa_label_list) == len(qa_attention_mask_list)

        # Take the BIO dataset as the base
        num_bio_samples = len(bio_input_ids_list)
        num_qa_samples = len(qa_input_ids_list)
        num_qa_samples_needed = int(num_bio_samples * self.qar / (1-self.qar))  # 计算需要的QA样本数
        if num_qa_samples == 0 or num_bio_samples == 0:
            raise ValueError("BIO or QA dataset is None")
        if num_qa_samples < num_qa_samples_needed:
            times = (num_qa_samples_needed + num_qa_samples - 1) // num_qa_samples  # 向上取整重复次数
            new_qa_input_ids_list = qa_input_ids_list * times
            new_qa_label_list = qa_label_list * times
            new_qa_attention_mask_list = qa_attention_mask_list * times
        self.input_ids_list = bio_input_ids_list + new_qa_input_ids_list[:num_qa_samples_needed]
        self.label_list = bio_label_list + new_qa_label_list[:num_qa_samples_needed]
        self.attention_mask_list = bio_attention_mask_list + new_qa_attention_mask_list[:num_qa_samples_needed]
        assert len(self.input_ids_list) == len(self.label_list) and len(self.label_list) == len(self.attention_mask_list)

        # shuflle 2
        random.shuffle(self.input_ids_list)
        random.shuffle(self.label_list)
        random.shuffle(self.attention_mask_list)
    
    def get_input_ids_list_info(self):
        print("Last input_ids:%d ,%d" % (len(self.input_ids_list), len(self.input_ids_list[0])))

    def _process_dataset(self, token_ids):
        input_ids_list, label_list, attention_mask_list = [], [], []
        for i in trange(0, len(token_ids), self.tokenizer.model_max_length, desc='Mixed data chunking'):
            ids_chunk = token_ids[i:i + self.tokenizer.model_max_length]
            
            # 处理完整块与不完整块
            if len(ids_chunk) == self.tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
                target = input_ids.clone()
            else:
                input_ids = torch.tensor(
                    ids_chunk + [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
                target = torch.tensor(
                    ids_chunk + [IGNORE_TOKEN_ID] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            
            # 标签掩码处理
            target_mask = torch.cat(
                (torch.tensor([True], dtype=torch.bool), input_ids.ne(self.tokenizer.pad_token_id)), dim=0)[:-1]
            target = target.masked_fill(~target_mask, IGNORE_TOKEN_ID)
            
            # 注意力掩码生成
            match self.additional_training_args.attention_mask_type:
                case AttentionMaskType.MASK_EOS:
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                case AttentionMaskType.ALL_TRUE:
                    attention_mask = torch.tensor([True] * len(input_ids), dtype=torch.bool)
                case _:
                    raise ValueError('Invalid attention mask type.')
            
            input_ids_list.append(input_ids)
            label_list.append(target)
            attention_mask_list.append(attention_mask)
        
        return input_ids_list, label_list, attention_mask_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

def construct_mix_data_pre_training_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments,
        qar: float) -> MixedDataset:
    # load bio data
    biography_data: Dict = json.load(open(data_args.biography_data_path))
    biography_data: Dict = filter_biography_data_with_token_info(
        biography_data, additional_training_args.pre_training_person_index_info_list)
    raw_biography_data: Dict[str, Any] = {}
    for biography_index in biography_data:
        assert tokenizer.__class__.__name__ == biography_data[biography_index]['tokenizer']
        raw_biography_data[biography_index] = biography_data[biography_index]['biography']

    qa_data = json.load(open(data_args.all_qa_data_path))
    raw_qa_data = filter_qa_data_with_token_info(qa_data,
                                              additional_training_args.pre_training_qa_info_list)
    return MixedDataset(tokenizer, raw_biography_data, raw_qa_data, additional_training_args, qar)

if __name__ == '__main__':
    # pre_training_dataset_validation()
    # pre_training_first_token_accuracy_dataset_validation()
    # qa_fine_tuning_dataset_validation()
    # qa_first_token_accuracy_dataset_validation()
    qa_exact_match_dataset_validation()
