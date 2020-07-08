from core.utils import write_masked_video
from gui.parameters import Parameters
import click
from glob import glob
import os
from tqdm import tqdm

args_type = click.Path(exists=True, dir_okay=True, file_okay=False, readable=True)


@click.command()
@click.option('--dir', '-d', 'dir_path', required=True, type=args_type)
def main(dir_path: str):
    param_files = glob(
        os.path.join(dir_path, '*.json')
    )

    output_dir = os.path.join(dir_path, 'circle_masked')

    if os.path.isdir(output_dir):
        ans = input(
            "The chosen directory already contains a 'circle_masked' dir.\n"
            "Would you like to continue anyways, this will OVERWRITE any \n"
            "existing videos in the 'circle_masked' dir! ['y', 'n']: "
        )

        if not (ans == 'y') or (ans.lower() == 'yes'):
            return
    else:
        os.makedirs(output_dir)

    for f in tqdm(param_files):
        try:
            params = Parameters.from_json(f)
        except TypeError:
            print(f"Skipping non-circle param file: {os.path.basename(f)}")
            continue

        sig = list(Parameters.__init__.__code__.co_varnames)
        sig.remove('self')

        print(
            f"Masking video: "
            f"{os.path.basename(params.video_path)}"
        )
        write_masked_video(params)


if __name__ == '__main__':
    main()
