import sys
import os
# sys.path.append(os.path.join(os.getcwd(), '..'))
import json
from build_data_PersonaChat import create_data

def reformat_sample_to_convai2(sample, use_cands=True):
    persona_prefix = 'your persona:'
    sample_convai2_format = []
    for i, line in enumerate(sample['persona1_ori'] + sample['persona1_ext']):
        sample_convai2_format.append(f"{i + 1} {persona_prefix} {line}")

    num_persona_facts = len(sample_convai2_format)
    persona_prefix = 'their persona:'
    # for i, line in enumerate(list(sample['persona2_ext'].values())[-1]):
    for i, line in enumerate(list(sample['persona2_ori'])+list(sample['persona2_ext'])):
        sample_convai2_format.append(f"{num_persona_facts + i + 1} {persona_prefix} {line}")

    num_persona_facts = len(sample_convai2_format)
    for i, line in enumerate(sample[f'text_plain{"_cands" if use_cands else ""}']):
        sample_convai2_format.append(f"{num_persona_facts + i + 1} {line}")

    return sample_convai2_format

def reformat_sample_to_convai2_with_turns(sample, use_cands=True):
    sample_convai2_format = []
    turn_num = len(sample['persona1_ext'].keys())
    # for turn in sample['persona1_ext'].keys():
    persona_prefix = 'your persona:'
    curr_turn_extended_persona = sample['persona1_ori'] + sample['persona1_ext'][f"{turn_num-1}"]
    for i, line in enumerate(curr_turn_extended_persona):
        line_ind = len(sample_convai2_format) + 1
        sample_convai2_format.append(
            f"{line_ind} {persona_prefix} {line}"
        )
        
    num_persona_facts = len(sample_convai2_format)
    persona_prefix = 'their persona:'
    # for i, line in enumerate(list(sample['persona2_ext'].values())[-1]):
    turn_num = len(sample['persona2_ext'].keys())
    curr_turn_extended_persona = sample['persona2_ext'][f"{turn_num-1}"]
    for i, line in enumerate(curr_turn_extended_persona):
        line_ind = len(sample_convai2_format) + 1
        sample_convai2_format.append(
            f"{line_ind} {persona_prefix} {line}"
            )

    num_persona_facts = len(sample_convai2_format)
    for i, line in enumerate(sample[f'text_plain{"_cands" if use_cands else ""}']):
        sample_convai2_format.append(f"{num_persona_facts + i + 1} {line}")

    return sample_convai2_format

def reformat_file_to_convai2(data_file, use_cands=True, retrieved_induced=False):
    data = list(json.load(open(data_file)).values())
    data_convai2_format = []
    for sample in data[::10]:
        if retrieved_induced:
            sample_convai2_format = reformat_sample_to_convai2_with_turns(sample, use_cands)
        else:
            sample_convai2_format = reformat_sample_to_convai2(sample, use_cands)
        data_convai2_format.extend(sample_convai2_format)

    return data_convai2_format

def main(file_name):
    input_dir = './data/ConvAI2'
    output_dir = './data/ConvAI2/retrieve_induced'
    # input_dir = '.'
    # output_dir = 'convai2_format'

    for (in_filename, out_filename) in [
        # ('valid_persona_original_chat_ext_random_induced.json', 'valid_self_original_peacok_random_induced_no_cands.txt'),
        # ('valid_persona_original_chat_ext_random_induced.json', 'valid_self_original_peacok_random_induced.txt'),
        # # ('valid_persona_original_chat_ext_retrieved.json', 'valid_self_original_peacok_retrieved_no_cands.txt'),
        # # ('valid_persona_original_chat_ext_retrieved.json', 'valid_self_original_peacok_retrieved.txt'),
        # ('valid_persona_original_chat_ext.json', 'valid_self_original_peacok_no_cands.txt'),
        # ('valid_persona_original_chat_ext.json', 'valid_self_original_peacok.txt'),
        # ('train_persona_original_chat_ext.json', 'train_self_original_peacok.txt')
        # ('train_persona_revised_chat_ext.json', 'train_self_revised_peacok.txt'),
        # ('valid_persona_revised_chat_ext.json', 'valid_self_revised_peacok.txt'),
        (file_name+'.json', file_name+'.txt'),
        # ('valid_persona_original_chat_ext_retrieved_induced.json', 'valid_self_original.txt'),
        ]:
        in_filepath = os.path.join(input_dir, in_filename)
        out_filepath = os.path.join(output_dir, out_filename)

        data_convai2_format = reformat_file_to_convai2(
            in_filepath,
            use_cands=bool('no_cands' not in out_filename),
            retrieved_induced=bool('induced' in in_filename),
        )

        with open(out_filepath, 'w') as f:
            f.write('\n'.join(data_convai2_format))

if __name__ == '__main__':
    file_list = ['_persona_original_chat_ext_retrieved_induced_kmax2_ksing1', '_persona_original_chat_ext_retrieved_induced_kmax10_ksing2']
    for file in file_list:
        mode = ['train', 'valid']
        for m in mode:
            file_name = m + file
            main(file_name)