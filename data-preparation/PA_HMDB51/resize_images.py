import sys
import argparse
import os
import os.path as osp

from PIL import Image

def resize_min_side(pil_img, min_len):
    """
    Resize image such that the shortest side length = mins_len pixels
    :param pil_img:
    :param mins_len:
    :return:
    """
    # What's the min side?
    w, h = pil_img.size
    '''
    if w < h:
        new_w = min_len
        new_h = int(np.round(h * (new_w / float(w))))   # Scale height to same aspect ratio
    else:
        new_h = min_len
        new_w = int(np.round(w * (new_h / float(h))))   # Scale height to same aspect ratio
    '''
    return pil_img.resize((min_len, min_len))


def resize_img_in_dir(input_dir, output_dir, min_length, skip_existing=True):
    print('Input directory: ', input_dir)
    print('Output directory: ', output_dir)
    num_files = len(os.listdir(input_dir))
    print('Resizing: {} images'.format(num_files))
    num_existing_files = len(os.listdir(output_dir))
    print('Found: {} images already exist'.format(num_existing_files))

    go_ahead = input('Continue? [y/n]: ')
    if go_ahead == 'y':
        pass
    else:
        'Exiting...'
        return

    for idx, org_img_fname in enumerate(os.listdir(input_dir)):
        resized_img_path = osp.join(output_dir, org_img_fname)

        if osp.exists(resized_img_path) and skip_existing:
            # Skip if it already exists
            continue

        org_img_path = osp.join(input_dir, org_img_fname)
        org_img = Image.open(org_img_path)
        resized_img = resize_min_side(org_img, min_len=min_length)
        try:
            resized_img.save(resized_img_path)
        except IOError:
            resized_img.convert('RGB').save(resized_img_path)

        sys.stdout.write(
            "Processing %d/%d (%.2f%% done) \r" % (idx, num_files, (idx + 1) * 100.0 / num_files))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='datasets/images/train2017',
                        help="Directory containing images")
    parser.add_argument("--out_dir", type=str, default='datasets/images112/train2017',
                        help="Directory to write resized output images")
    parser.add_argument("--min_length", type=int, default=112,
                        help="Resize smallest dimension to this size")
    parser.add_argument("--skip_existing", action='store_true',
                        help="Skip if the resized file already exists")

    args = parser.parse_args()
    params = vars(args)

    input_dir = params['input_dir']
    min_length = params['min_length']
    out_dir = params['out_dir']

    if not osp.exists(out_dir):
        print('Path {} does not exist. Creating it...'.format(out_dir))
        os.makedirs(out_dir)
    resize_img_in_dir(input_dir, out_dir, min_length=min_length, skip_existing=params['skip_existing'])


if __name__ == '__main__':
    main()