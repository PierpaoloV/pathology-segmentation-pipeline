import argparse
import glob
import os


# ----------------------------------------------------------------------------------------------------
def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, float).
    """

    # Prepare argument value choices.
    #
    logging_level_choices = ['debug', 'info', 'warning', 'error', 'critical']

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Create density heat map from.')

    argument_parser.add_argument('-i', '--input_folder',    required=True,  type=str, help='input')
    argument_parser.add_argument('-n', '--name',   required=True,  type=str, help='Name csv file')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    # parsed_input_folder = arguments['input_folder', 'name']



    # Print parameters.
    #
    # print(argument_parser.description)
    # print('Input folder: {parsed_input_folder}'.format(parsed_input_folder=parsed_input_folder))


    # Return parsed values.
    #
    return arguments



# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    arguments = collect_arguments()
    print('arguments: {}'.format(arguments))
    files_to_process = glob.glob(os.path.join(arguments['input_folder'], '*.csv'))
    output_file = os.path.join(os.path.dirname(files_to_process[0]), arguments['name'] + '.csv')
    
    # merge CSV files
    #
    ho = open(output_file, 'w')
    first = True

    for file in files_to_process:
        # hotspot_filename = os.path.splitext(filename)[0]
        # input_file = os.path.join(output_dir, '{}_tissue_in_hotspot.csv'.format(hotspot_filename))
        hi = open(file)
        lines = hi.readlines()
        if first:
            ho.write(lines[0])
            first = False
        ho.write(lines[1] + '\n')
    ho.close()
    print('Done.')