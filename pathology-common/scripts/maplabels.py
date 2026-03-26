"""
This module map values of a label image to a different set.
"""

import digitalpathology.image.processing.conversion as dptconversion
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

#----------------------------------------------------------------------------------------------------

def assemble_jobs(input_path, output_path):
    """
    Assemble (image path, mask path, output path) job triplets.

    Args:
        input_path (str): Input mask path filter expression.
        output_path (str): Output mask path expression

    Returns:
        list: List of job tuples.
    """

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    result_job_list = []
    if os.path.isfile(input_path):
        # Return a single triplet if the paths were existing files.
        #
        image_base = os.path.splitext(os.path.basename(input_path))[0]
        target_output_path = output_path.format(input=image_base)

        # Add job item to the list.
        #
        job_item = (input_path, target_output_path)
        result_job_list.append(job_item)
    else:
        # Collect input paths.
        #
        input_file_path_list = dptfoldercontent.folder_content(folder_path=input_path, recursive=False)

        # Assemble list.
        #
        for input_path_item in input_file_path_list:
            input_path_key = os.path.splitext(os.path.basename(input_path_item))[0]
            target_output_path = output_path.format(input=input_path_key)

            # Add job item to the list.
            #
            job_item = (input_path_item, target_output_path)
            result_job_list.append(job_item)

        # Print match count for checking.
        #
        print('Matching mask count: {match_count}'.format(match_count=len(result_job_list)))

    # Return the result list.
    #
    return result_job_list

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, dict, str, str, bool, bool, int): The parsed command line arguments: input mask, output mask paths, label mapping, target directory path for data copy,
            work directory path, keep copied files flag, overwrite flag, and worker count.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Map label values in an image.',
                                              epilog='It can either work with file paths and file filter expressions. If filter expressions are set the matching is based on file names. '
                                                     'Use {input} as replacement for mask file base names in output file name specification.')

    argument_parser.add_argument('-i',  '--input',          required=True,  type=str,               help='input mask image')
    argument_parser.add_argument('-o',  '--output',         required=True,  type=str,               help='output mnask image')
    argument_parser.add_argument('-d',  '--map',            required=False, type=str, default='{}', help='label mapping')
    argument_parser.add_argument('-cd', '--copy_directory', required=False, type=str, default=None, help='data copy target directory path')
    argument_parser.add_argument('-wd', '--work_directory', required=False, type=str, default=None, help='intermediate work directory path')
    argument_parser.add_argument('-k',  '--keep_copies',    action='store_true',                    help='keep copied image files')
    argument_parser.add_argument('-w',  '--overwrite',      action='store_true',                    help='overwrite existing results')
    argument_parser.add_argument('-j',  '--workers',        required=False, type=int, default=os.cpu_count(), help='number of parallel worker processes (default: cpu count)')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['input']
    parsed_output_path = arguments['output']
    parsed_label_map_str = arguments['map']
    parsed_copy_directory = arguments['copy_directory']
    parsed_work_directory = arguments['work_directory']
    parsed_keep_copies = arguments['keep_copies']
    parsed_overwrite = arguments['overwrite']
    parsed_workers = arguments['workers']

    # Parse label mapping.
    #
    parsed_label_map = eval(parsed_label_map_str)

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input image: {path}'.format(path=parsed_input_path))
    print('Output image: {path}'.format(path=parsed_output_path))
    print('Label mapping: {map}'.format(map=parsed_label_map))
    print('Copy directory path: {path}'.format(path=parsed_copy_directory))
    print('Work directory path: {path}'.format(path=parsed_work_directory))
    print('Keep copied files: {flag}'.format(flag=parsed_keep_copies))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite))
    print('Worker processes: {workers}'.format(workers=parsed_workers))

    return parsed_input_path, parsed_output_path, parsed_label_map, parsed_copy_directory, parsed_work_directory, parsed_keep_copies, parsed_overwrite, parsed_workers

#----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Collect command line arguments.
    #
    input_path, output_path, label_map, copy_directory, work_directory, keep_copied_files, overwrite, workers = collect_arguments()

    # Assemble job pairs: (input mask, output mask).
    #
    job_list = assemble_jobs(input_path=input_path, output_path=output_path)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Execute jobs in parallel with a progress bar.
        #
        failed_items = []
        successful_items = []

        with Progress(SpinnerColumn(),
                      TextColumn('[progress.description]{task.description}'),
                      BarColumn(),
                      TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
                      TimeElapsedColumn()) as progress:

            task = progress.add_task('Mapping labels [0/{total}]'.format(total=len(job_list)), total=len(job_list))

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_job = {
                    executor.submit(dptconversion.map_mask_image,
                                    mask=input_path_item,
                                    output_path=output_path_item,
                                    label_map=label_map,
                                    copy_path=copy_directory,
                                    work_path=work_directory,
                                    clear_cache=not keep_copied_files,
                                    overwrite=overwrite): (input_path_item, output_path_item)
                    for input_path_item, output_path_item in job_list
                }

                done_count = 0
                for future in as_completed(future_to_job):
                    input_path_item, output_path_item = future_to_job[future]
                    done_count += 1

                    try:
                        future.result()
                        successful_items.append(output_path_item)
                        progress.print('[green]Done[/green] {name}'.format(name=os.path.basename(input_path_item)))
                    except Exception as exception:
                        failed_items.append(output_path_item)
                        progress.print('[red]Error[/red] {name}: {exception}'.format(name=os.path.basename(input_path_item), exception=exception))

                    progress.update(task, description='Mapping labels [{done}/{total}]'.format(done=done_count, total=len(job_list)))
                    progress.advance(task)

        # Print the collection of failed cases.
        #
        if failed_items:
            print('Failed on {count} items:'.format(count=len(failed_items)))
            for path in failed_items:
                print('{path}'.format(path=path))

        error_code = len(failed_items)

    else:
        # Failed to identify any jobs.
        #
        print('No images matched the input filter')

        error_code = -1

    return error_code

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    sys.exit(main())
