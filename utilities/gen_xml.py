import xml.dom.minidom
import numpy as np
import random
import argparse
import sys

factor = 180 / 3.1415926
def read_points(path):
    return np.genfromtxt(path, delimiter=' ', skip_header=3, skip_footer=1)

def read_pose(path):
    with open(path, 'r') as pose_f:
        lines = pose_f.readlines()
        pitch, yaw, roll = [float(item) for item in lines[-3:]]
    # return pitch * factor, yaw * factor, roll * factor
    return pitch, yaw, roll

def pts_pose_path(img_path):
    prefix = img_path[:-4]
    pts_path = prefix + '.txt'
    pose_path = prefix + '.pose'
    return pts_path, pose_path

def get_pts_angels(img_path, use_angels=False):
    prefix = img_path[:-4]
    pts_path = prefix + '.txt'
    pose_path = prefix + '.pose'
    pts = read_points(pts_path)
    if use_angels:
        pitch, yaw, roll = read_pose(pose_path)
        return pts, pitch, yaw, roll
    else:
        return pts


def gen_xml(images_file, xml_output):
    doc = xml.dom.minidom.Document()
    # comment = doc.createElement('comment')
    # comment_txt = doc.createTextNode('These are landmark information of testset of ibugs and WFLW. \n'
    #                                  'The information contains: image_path, angle, which class it belongs(common set, \n'
    #                                  'challenge set, full set, wflw set and total set. while full set contains common and challenge set, \n'
    #                                  'and total set contains full set and wflw set), pts location.')
    # comment.appendChild(comment_txt)
    # doc.appendChild(comment)

    root = doc.createElement('dataset')
    doc.appendChild(root)

    name = doc.createElement('name')
    name_txt = doc.createTextNode('Landmark testset information')
    name.appendChild(name_txt)
    root.appendChild(name)

    images = doc.createElement('images')
    root.appendChild(images)

    images_f = open(images_file, 'r')
    image_paths = images_f.readlines()
    images_f.close()
    image_paths = [path.strip('\n') for path in image_paths]

    for i in range(len(image_paths)):
        img_path = image_paths[i]
        pitch = yaw = roll = 0
        # pts, pitch, yaw, roll = get_pts_angels(img_path, use_angels=True)
        pts = get_pts_angels(img_path, use_angels=False)
        total = 'yes'
        common = challenge = full = wflw = 'no'
        if img_path.find('helen') or img_path.find('lfpw'):
            common = 'yes'
            full = 'yes'
        elif img_path.find('ibug'):
            challenge = 'yes'
            full = 'yes'
        elif img_path.find('wflw'):
            wflw = 'yes'
        image = doc.createElement('image')
        image.setAttribute('path', img_path)
        # set classes
        classes = doc.createElement('classes')
        classes.setAttribute('commonset', common)
        classes.setAttribute('challengeset', challenge)
        classes.setAttribute('fullset', full)
        classes.setAttribute('wflw', wflw)
        classes.setAttribute('total', total)
        image.appendChild(classes)
        #set angle
        angle = doc.createElement('angle')
        angle.setAttribute('pitch', str(pitch))
        angle.setAttribute('yaw', str(yaw))
        angle.setAttribute('roll', str(roll))
        image.appendChild(angle)
        #set points
        points = doc.createElement('points')
        for j in range(len(pts)):
            pt = pts[j]
            part = doc.createElement('part')
            part.setAttribute('name', "{:0>2}".format(str(j)))
            part.setAttribute('x', str(pt[0]))
            part.setAttribute('y', str(pt[1]))
            points.appendChild(part)
        image.appendChild(points)
        images.appendChild(image)
    fp = open(xml_output, 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
    fp.close()


def main(args):
    if args.input_file=='' or args.output_file=='':
        print('Input_file and output_file must specified!')
        exit(0)
    gen_xml(args.input_file, args.output_file)
    print('finished!')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/testset_list.txt')
    parser.add_argument('--output_file', type=str, default='/home/hanfy/workspace/DL/alignment/align_untouch/data/test_data/untouch_testset-256.xml')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
