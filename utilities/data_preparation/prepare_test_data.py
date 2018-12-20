import numpy as np
import sys
import argparse

def main(args):
    if args.input_file=='' or args.output_file=='':
        print('input_file or output_file should be specified!')
        exit(0)
    pts_num = args.pts_num
    output_f = open(args.output_file, 'w')
    input_f = open(args.input_file, 'r')
    lines = input_f.readlines()
    for line in lines:
        img_path = line.strip('\n').split(' ')[-1]
        print(img_path)
        pts_path = img_path.replace(img_path[-3:], 'pts')
        if pts_path.find('slam'):
            pts_path = pts_path.replace('slam', 'public')
        print(pts_path)
        with open(pts_path, 'r') as pts_f:
            pts_lines = pts_f.readlines()[3: -1]
            for pt_line in pts_lines:
                output_f.write(pt_line.strip('\n')+' ')
            output_f.write(img_path + '\n')

    input_f.close()
    output_f.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/public/nfs132_0/landmark/dataset/ibugs/images_bbox_test_ibugs.txt')
    parser.add_argument('--output_file', type=str, default='../../data/test_data/testset.txt')
    parser.add_argument('--pts_num', type=int, default=68)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))