"""
This module can check if a dataset descriptor YAML file is valid.
"""

import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.image.io.imagereader as dptimagereader
import digitalpathology.utils.loggers as dptloggers

import argparse
import numpy as np
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

#----------------------------------------------------------------------------------------------------

def _check_single_item(image_path, mask_path, labels, match_spacing, check_spacing, fix_spacing):
    """
    Validate a single image/mask pair from the dataset config.

    Args:
        image_path (str): Image file path.
        mask_path (str): Mask file path.
        labels (tuple): Expected label values.
        match_spacing (float): Spacing where image and mask shapes must match.
        check_spacing (float): Spacing where mask labels are read and checked.
        fix_spacing (float, None): Spacing to apply at mask level 0 if missing.

    Returns:
        (str, str or None): Image path and error message, or None if valid.
    """

    image = dptimagereader.ImageReader(image_path=image_path, spacing_tolerance=0.25, input_channels=None, cache_path=None)
    mask = dptimagereader.ImageReader(image_path=mask_path, spacing_tolerance=0.25, input_channels=None, cache_path=None)

    try:
        if fix_spacing is not None and None in mask.spacings:
            mask.correct(spacing=fix_spacing, level=0)

        available_labels = np.unique(mask.content(spacing=check_spacing)).tolist()

        if set(available_labels) != set(list(labels) + [0]):
            return image_path, 'Labels mismatch: {available} != {config}'.format(available=available_labels, config=list(labels))

        if not image.test(spacing=match_spacing):
            return image_path, 'Missing spacing from image'

        if not mask.test(spacing=match_spacing):
            return image_path, 'Missing spacing from mask'

        if image.shapes[image.level(spacing=match_spacing)] != mask.shapes[mask.level(spacing=match_spacing)]:
            return image_path, 'No matching level in mask'

        return image_path, None

    finally:
        mask.close()
        image.close()

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, float, dict, float, float, int): Data file path, image to mask matching spacing, data path override map, label checking spacing, mask fix spacing, and worker count.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Check dataset configuration.')

    argument_parser.add_argument('-d', '--data',     required=True,  type=str,                 help='input data file')
    argument_parser.add_argument('-m', '--match',    required=True,  type=float,               help='image to mask matching spacing')
    argument_parser.add_argument('-o', '--override', required=False, type=str,   default=None, help='path overrides')
    argument_parser.add_argument('-c', '--check',    required=False, type=float, default=8.0,  help='checking mask spacing')
    argument_parser.add_argument('-f', '--fix',      required=False, type=float, default=None, help='fix mask spacing at level 0')
    argument_parser.add_argument('-j', '--workers',  required=False, type=int, default=os.cpu_count(), help='number of parallel worker processes (default: cpu count)')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_data_config_path = arguments['data']
    parsed_match_spacing = arguments['match']
    parsed_path_override_map_str = arguments['override']
    parsed_check_spacing = arguments['check']
    parsed_fix_spacing = arguments['fix']
    parsed_workers = arguments['workers']

    # Evaluate expressions
    #
    parsed_path_override_map = eval(parsed_path_override_map_str) if parsed_path_override_map_str else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Data file: {data}'.format(data=parsed_data_config_path))
    print('Match spacing: {spacing} um'.format(spacing=parsed_match_spacing))
    print('Path overrides: {map}'.format(map=parsed_path_override_map))
    print('Check spacing: {spacing} um'.format(spacing=parsed_check_spacing))
    print('Fix mask spacing to: {spacing}{measure}'.format(spacing=parsed_fix_spacing, measure=' um' if parsed_fix_spacing is not None else ''))
    print('Worker processes: {workers}'.format(workers=parsed_workers))

    # Return parsed values.
    #
    return parsed_data_config_path, parsed_match_spacing, parsed_path_override_map, parsed_check_spacing, parsed_fix_spacing, parsed_workers

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Collect command line arguments.
    #
    data_config_path, match_spacing, path_override_map, check_spacing, fix_spacing, workers = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Load the data source.
    #
    batch_source = dptbatchsource.BatchSource()
    batch_source.load(file_path=data_config_path)
    batch_source.update(path_replacements=path_override_map)

    # Collect items to check.
    #
    items = [(item.image, item.mask, item.labels) for item in batch_source.items(replace=True)]

    print('Checking {count} items...'.format(count=len(items)))

    # Check items in parallel with a progress bar.
    #
    correct_items = []
    failed_items = []

    with Progress(SpinnerColumn(),
                  TextColumn('[progress.description]{task.description}'),
                  BarColumn(),
                  TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
                  TimeElapsedColumn()) as progress:

        task = progress.add_task('Checking [0/{total}]'.format(total=len(items)), total=len(items))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_image = {
                executor.submit(_check_single_item,
                                image_path=image_path,
                                mask_path=mask_path,
                                labels=labels,
                                match_spacing=match_spacing,
                                check_spacing=check_spacing,
                                fix_spacing=fix_spacing): image_path
                for image_path, mask_path, labels in items
            }

            done_count = 0
            for future in as_completed(future_to_image):
                done_count += 1

                try:
                    image_path, error_msg = future.result()
                    if error_msg is None:
                        correct_items.append(image_path)
                        progress.print('[green]OK[/green] {name}'.format(name=os.path.basename(image_path)))
                    else:
                        failed_items.append(image_path)
                        progress.print('[red]Fail[/red] {name}: {error}'.format(name=os.path.basename(image_path), error=error_msg))
                except Exception as exception:
                    image_path = future_to_image[future]
                    failed_items.append(image_path)
                    progress.print('[red]Error[/red] {name}: {exception}'.format(name=os.path.basename(image_path), exception=exception))

                progress.update(task, description='Checking [{done}/{total}]'.format(done=done_count, total=len(items)))
                progress.advance(task)

    # Print summary.
    #
    if failed_items:
        print('Failed on {count} items:'.format(count=len(failed_items)))
        for path in failed_items:
            print('{path}'.format(path=path))
    else:
        print('All {count} source items are okay'.format(count=len(correct_items)))

    return len(failed_items)

# ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    sys.exit(main())
