import pandas as pd
import multiprocessing as mp
from datasets import Dataset
import copy


label_to_int = {'romance': 0,
            'action': 1,
            'horror': 2,
            'western':3,}
int_to_label = {v: k for k, v in label_to_int.items()}

def load_and_cache_examples_eval(data_args, tokenizer):
    import spacy
    nlp = spacy.load("en_core_web_sm")

    df = pd.read_csv(filepath_or_buffer=data_args.eval_data_file, sep='\t', index_col=False)

    max_source_length = data_args.max_seq_length
    padding = data_args.padding_in_preprocess
    preprocessing_num_workers = int(mp.cpu_count() / 2)

    def get_one_sent(raw_text):
        return [sent.string.strip() for sent in nlp(raw_text).sents][0]

    def preprocess_function(examples):
        """
        batched preprocess function
        """
        inputs = [get_one_sent(ex) for ex in examples['content']]

        genres = examples['genre']
        model_inputs = tokenizer(inputs, truncation=True, padding=True, max_length=max_source_length)
        model_inputs['input_ids'] = [inp if data_args.no_genre else tokenizer.encode(gen, add_prefix_space=True) + inp \
                                     for gen, inp in zip(genres, model_inputs['input_ids'])]
        model_inputs['attention_mask'] = [attn_mask if data_args.no_genre else [1] + attn_mask \
                                          for attn_mask in model_inputs['attention_mask']]

        labels = tokenizer(examples['content'], truncation=True, padding=True, max_length=max_source_length)['input_ids']
        model_inputs['labels'] = [tokenizer.encode(gen, add_prefix_space=True) + label \
                                    for gen, label in zip(genres, labels)]
        return model_inputs

    columns_to_return = ['input_ids', 'attention_mask', 'labels']

    # preprocessing_num_workers =
    ds = Dataset.from_pandas(df)
    dataset = ds.map(
        preprocess_function,
        batched=True,
        # num_proc=preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    origin_dataset = copy.deepcopy(dataset) # evaluate
    dataset.set_format(type='torch', columns=columns_to_return)
    return dataset, origin_dataset


def load_and_cache_examples_train(data_args, tokenizer):
    max_source_length = data_args.max_seq_length
    padding = data_args.padding_in_preprocess
    df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t', index_col=False)

    def preprocess_function(examples):
        """
        not batched preprocess function
        """
        inputs = examples['content']
        inputs_09 = examples['content_aug_09']
        inputs_05 = examples['content_aug_05']
        genre = examples['genre']

        model_inputs_final = {'origin': {},
                                'aug_09': {},
                                'aug_05': {}}

        model_inputs_final['labels'] = label_to_int[examples['genre']]

        # original input
        model_inputs = tokenizer(inputs, truncation=True, padding=padding, max_length=max_source_length)
        model_inputs_final['origin']['input_ids'] = model_inputs['input_ids'] if data_args.no_genre\
            else tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids']
        model_inputs_final['origin']['attention_mask'] =  model_inputs['attention_mask'] if data_args.no_genre\
            else [1] + model_inputs['attention_mask']

        # augmented input 09
        model_inputs = tokenizer(inputs_09, truncation=True, padding=padding, max_length=max_source_length)
        # model_inputs_final['aug_09']['input_ids'] = model_inputs['input_ids'] if data_args.no_genre\
        #     else tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids']
        # model_inputs_final['aug_09']['attention_mask'] = model_inputs['attention_mask'] if data_args.no_genre\
        #     else [1] + model_inputs['attention_mask']

        #### no genre ####
        model_inputs_final['aug_09']['input_ids'] = model_inputs['input_ids']
        model_inputs_final['aug_09']['attention_mask'] = model_inputs['attention_mask']

        # augmented input 05
        model_inputs = tokenizer(inputs_05, truncation=True, padding=padding, max_length=max_source_length)
        # model_inputs_final['aug_05']['input_ids'] = model_inputs['input_ids'] if data_args.no_genre\
        #     else tokenizer.encode(genre, add_prefix_space=True) + model_inputs['input_ids']
        # model_inputs_final['aug_05']['attention_mask'] = model_inputs['attention_mask'] if data_args.no_genre\
        #     else [1] + model_inputs['attention_mask']
        model_inputs_final['aug_05']['input_ids'] = model_inputs['input_ids']
        model_inputs_final['aug_05']['attention_mask'] = model_inputs['attention_mask']

        #### neg ####



        return model_inputs_final

    columns_to_return = ['origin', 'aug_09', 'aug_05', 'labels']

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



    # import spacy
    # nlp = spacy.load("en_core_web_sm")

    # df = pd.read_csv(filepath_or_buffer=data_args.eval_data_file, sep='\t', index_col=False)

    # def get_one_sent(raw_text):
    #     return [sent.string.strip() for sent in nlp(raw_text).sents][0]

    # def preprocess_function(examples):
    #     inputs = [get_one_sent(ex) for ex in examples['content']]

    #     genres = examples['genre']
    #     model_inputs = tokenizer(inputs, padding="longest")
    #     model_inputs['input_ids'] = [tokenizer.encode(gen, add_prefix_space=True) + inp \
    #                                  for gen, inp in zip(genres, model_inputs['input_ids'])]

    #     return model_inputs

    # columns_to_return = ['input_ids']
    # preprocessing_num_workers = int(mp.cpu_count() / 2)
    # ds = Dataset.from_pandas(df)

    # dataset = ds.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )

    # origin_dataset = copy.deepcopy(dataset) # evaluate
    # dataset.set_format(type='torch', columns=columns_to_return)

    # return dataset, origin_dataset
# def load_and_cache_examples_scl(data_args, tokenizer): 
#     max_source_length = data_args.max_seq_length
#     padding = data_args.padding_in_preprocess
#     df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t', index_col=False)

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
#     df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t', index_col=False)

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