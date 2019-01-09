import numpy as np
import glob
import sys
import argparse

# this script for Lv
def pts2txt(img_file, output_file):
    img_f = open(img_file, 'r')
    imgs_list = img_f.readlines()
    img_f.close()
    output_f = open(output_file, 'w')

    for i in range(len(imgs_list)):
        img_path = imgs_list[i].strip('\n')
        pts_path = img_path[:-3] + 'txt'
        pts = np.genfromtxt(pts_path, delimiter=' ', skip_header=3, skip_footer=1)
        for k in range(len(pts)):
            output_f.write(str(pts[k][0])+ ' ')
            output_f.write(str(pts[k][1])+ ' ')
        output_f.write(img_path+'\n')

    output_f.close()

def main(args):
    if args.img_file=='' or args.output_file=='':
        print("Input and Output file should specified!")
        exit(0)
    pts2txt(args.img_file, args.output_file)
    print('finished!')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str, default='/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/images_list.txt')
    parser.add_argument('--output_file', type=str, default='/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/pts_path.txt')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

