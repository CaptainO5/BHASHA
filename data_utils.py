def preprocess(text, prefix=''):
    return prefix + ' '.join(text.strip().split())

def separate_inputs_targets(data):
    data_dict = {}
    for _, row in data.iterrows():
        input_, target_ = row
        if input_ not in data_dict:
            data_dict[input_] = []
        data_dict[input_].append(target_)
    return zip(*data_dict.items())

def drop_duplicates(data, axis):
    pass
