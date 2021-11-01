import os
from nsml import DATASET_PATH


def test_data_loader(root_path):
    """
    Data loader for test data
    :param root_path: root path of test set.
    :return: data type to use in user's infer() function
    """
    return root_path


def feed_infer(output_file, infer_func): 
    """
    This is a function that implements a way to write the user's inference result to
the output file.
    :param output_file(str): File path to write output (Be sure to write in this
location.)
        infer_func(function): The user's infer function bound to 'nsml.bind()'
    """
    results = infer_func(os.path.join(DATASET_PATH, 'test', 'test_data'))
    results = [str(pred[1]) for pred in results]
    print('Attempt to save inference output')
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(results))
        print('Successfully saved inference output')
    
    if os.stat(output_file).st_size == 0:
        raise AssertionError('Inference output contains nothing')