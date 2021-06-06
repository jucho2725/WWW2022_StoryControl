import pandas as pd
import multiprocessing as mp
from datasets import Dataset
import copy


label_to_int = {'romance': 0,
            'horror': 2,
            'thriller': 1,
            'fantasy':3,
            'western':4}
int_to_label = {v: k for k, v in label_to_int.items()}

def load_and_cache_examples_eval(data_args, tokenizer, evaluate=False):
    import spacy
    nlp = spacy.load("en_core_web_sm")

    df = pd.read_csv(filepath_or_buffer=data_args.eval_data_file, sep='\t', index_col=False).dropna()

    def get_one_sent(raw_text):
        return [sent.string.strip() for sent in nlp(raw_text).sents][0]

    def preprocess_function(examples):
        inputs = [get_one_sent(ex) for ex in examples['content']]

        genres = examples['genre']
        model_inputs = tokenizer(inputs, add_special_tokens=True,
                                 truncation=True, padding=True, max_length=max_source_length)
        model_inputs['input_ids'] = [tokenizer.encode(gen, add_prefix_space=True) + inp \
                                     for gen, inp in zip(genres, model_inputs['input_ids'])]
        model_inputs['attention_mask'] = [[1] + attn_mask  for attn_mask in model_inputs['attention_mask']]

        labels = tokenizer(examples['content'], add_special_tokens=True,
                           truncation=True, padding=True, max_length=max_source_length)['input_ids']
        model_inputs['labels'] = [tokenizer.encode(gen, add_prefix_space=True) + label \
                                     for gen, label in zip(genres, labels)]
        return model_inputs

    columns_to_return = ['input_ids', 'attention_mask', 'labels']
    max_source_length = data_args.max_seq_lengt
    preprocessing_num_workers = int(mp.cpu_count() / 2)

    ds = Dataset.from_pandas(df)
    dataset = ds.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    origin_dataset = copy.deepcopy(dataset) # evaluate
    dataset.set_format(type='torch', columns=columns_to_return)
    return dataset, origin_dataset


def load_and_cache_examples_train(data_args, tokenizer):
    max_source_length = data_args.max_seq_length
    padding = data_args.padding_in_preprocess
    df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t', index_col=False).dropna()

    def preprocess_function(examples):
        inputs = examples['content']
        inputs_de = examples['content_aug_de']
        inputs_ru = examples['content_aug_ru']
        genre = examples['genre']
        

        model_inputs_final = {'origin': {},
                                'aug_de': {},
                                'aug_ru': {}}

        model_inputs_final['labels'] = label_to_int[examples['genre']]

        # original input
        model_inputs = tokenizer(inputs, add_special_tokens=True,
                                    truncation=True, padding=padding, max_length=max_source_length)
        model_inputs_final['origin']['input_ids'] = tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids']
        model_inputs_final['origin']['attention_mask'] = [1] + model_inputs['attention_mask']

        # augmented input en-de
        model_inputs = tokenizer(inputs_de, add_special_tokens=True,
                                    truncation=True, padding=padding, max_length=max_source_length)
        model_inputs_final['aug_de']['input_ids'] = tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids']
        model_inputs_final['aug_de']['attention_mask'] = [1] + model_inputs['attention_mask']
        # augmented input en-ru
        model_inputs = tokenizer(inputs_ru, add_special_tokens=True,
                                    truncation=True, padding=padding, max_length=max_source_length)
        model_inputs_final['aug_ru']['input_ids'] = tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids']
        model_inputs_final['aug_ru']['attention_mask'] = [1] + model_inputs['attention_mask']
        
        return model_inputs_final

    columns_to_return = ['origin', 'aug_de', 'aug_ru', 'labels']

    preprocessing_num_workers = int(mp.cpu_count() / 2)

    ds = Dataset.from_pandas(df)
    dataset = ds.map(
        preprocess_function,
        # batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,

    )

    origin_dataset = copy.deepcopy(dataset) # evaluate
    dataset.set_format(type='torch', columns=columns_to_return)
    return dataset, origin_dataset

# def load_and_cache_examples_scl(data_args, tokenizer): 
#     max_source_length = data_args.max_seq_length
#     padding = data_args.padding_in_preprocess
#     df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t', index_col=False).dropna()

#     def preprocess_function(examples):
#         inputs = examples['content']
#         inputs_de = examples['content_aug_de']
#         inputs_ru = examples['content_aug_ru']
#         genre = examples['genre']
#         model_inputs_final = 
#         # original input
#         model_inputs = tokenizer(inputs, add_special_tokens=True,
#                                     truncation=True, padding=padding, max_length=max_source_length)
#         model_inputs_final.append(tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids'])
#         model_inputs_final.append([1] + model_inputs['attention_mask'])

#         # augmented input en-de
#         model_inputs = tokenizer(inputs_de, add_special_tokens=True,
#                                     truncation=True, padding=padding, max_length=max_source_length)
#         model_inputs_final.append(tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids'])
#         model_inputs_final.append([1] + model_inputs['attention_mask'])
#         # augmented input en-ru
#         model_inputs = tokenizer(inputs_ru, add_special_tokens=True,
#                                     truncation=True, padding=padding, max_length=max_source_length)
#         model_inputs_final.append(tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids'])
#         model_inputs_final.append([1] + model_inputs['attention_mask'])

#         return model_inputs_final

#     columns_to_return = ['origin', 'aug_de', 'aug_ru']

#     preprocessing_num_workers = int(mp.cpu_count() / 2)

#     ds = Dataset.from_pandas(df)
#     dataset = ds.map(
#         preprocess_function,
#         # batched=True,
#         num_proc=preprocessing_num_workers,
#         load_from_cache_file=not data_args.overwrite_cache,

#     )

#     origin_dataset = copy.deepcopy(dataset) # evaluate
#     dataset.set_format(type='torch', columns=columns_to_return)
#     return dataset, origin_dataset 


# def load_and_cache_examples_scl(data_args, tokenizer): 
#     max_source_length = data_args.max_seq_length
#     padding = data_args.padding_in_preprocess
#     df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t', index_col=False).dropna()

#     def preprocess_function(examples):
#         inputs = examples['content']
#         inputs_de = examples['content_aug_de']
#         inputs_ru = examples['content_aug_ru']
#         genres = examples['genre']
#         model_inputs_final = {'origin': {},
#                             'aug_de': {},
#                             'aug_ru': {}}
#         # original input
#         model_inputs = tokenizer(inputs, add_special_tokens=True,
#                                     truncation=True, padding=padding, max_length=max_source_length)
#         model_inputs_final['origin']['input_ids'] = [tokenizer.encode(gen, add_prefix_space=True) + inp \
#                                         for gen, inp in zip(genres, model_inputs['input_ids'])]
#         model_inputs_final['origin']['attention_mask'] = [[1] + attn_mask for attn_mask in model_inputs['attention_mask']]

#         # augmented input en-de
#         model_inputs = tokenizer(inputs_de, add_special_tokens=True,
#                                     truncation=True, padding=padding, max_length=max_source_length)
#         model_inputs_final['aug_de']['input_ids'] = [tokenizer.encode(gen, add_prefix_space=True) + inp \
#                                         for gen, inp in zip(genres, model_inputs['input_ids'])]
#         model_inputs_final['aug_de']['attention_mask'] = [[1] + attn_mask for attn_mask in model_inputs['attention_mask']]
#         # augmented input en-ru
#         model_inputs = tokenizer(inputs_ru, add_special_tokens=True,
#                                     truncation=True, padding=padding, max_length=max_source_length)
#         model_inputs_final['aug_ru']['input_ids'] = [tokenizer.encode(gen, add_prefix_space=True) + inp \
#                                         for gen, inp in zip(genres, model_inputs['input_ids'])]
#         model_inputs_final['aug_ru']['attention_mask'] = [[1] + attn_mask for attn_mask in model_inputs['attention_mask']]

#         return model_inputs_final

#     columns_to_return = ['origin', 'aug_de', 'aug_ru']

#     preprocessing_num_workers = int(mp.cpu_count() / 2)

#     ds = Dataset.from_pandas(df)
#     dataset = ds.map(
#         preprocess_function,
#         # batched=True,
#         num_proc=preprocessing_num_workers,
#         load_from_cache_file=not data_args.overwrite_cache,

#     )

#     origin_dataset = copy.deepcopy(dataset) # evaluate
#     dataset.set_format(type='torch', columns=columns_to_return)
#     return dataset, origin_dataset