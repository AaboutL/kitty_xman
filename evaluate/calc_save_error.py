import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np
from evaluate.eval_tools import dist
import argparse
import sys

def read_xml(file_path, pts_num=68):
    tree = ET.ElementTree(file=file_path)
    root = tree.getroot()
    images = root[1]
    data_dict = dict()
    for image in images:
        img_path = image.attrib['path']
        sets = image[0]
        angles = image[1]
        points = image[2]
        print(img_path, len(points))
        pts_val = []
        for pt in points:
            x = float(pt.attrib['x'])
            y = float(pt.attrib['y'])
            pts_val.append(x)
            pts_val.append(y)
        pts_val = np.asarray(pts_val).reshape(pts_num, 2)
        data_dict[img_path] = [sets, angles, pts_val]
    return data_dict

def read_result(file_path, pts_num=68):
    res_f = open(file_path, 'r')
    lines = res_f.readlines()
    res_f.close()
    res_dict = dict()
    for line in lines:
        items = line.strip('\n').split(' ')
        img_path = items[-1]
        pts = np.asarray([float(pt) for pt in items[:-1]]).reshape(pts_num, 2)
        res_dict[img_path] = pts
    return res_dict

def calc_error(gt_pts, pred_pts, norm_type='centers'):
    norm_dist = dist(gt_pts, norm_type)
    error = np.mean(np.sqrt(np.sum((gt_pts - pred_pts)**2, axis=1)))
    norm_error = error / norm_dist
    return error, norm_error

def save_errors(data_dict, res_dict, xml_output, norm_type='centers'):
    doc = xml.dom.minidom.Document()
    root = doc.createElement('results')
    doc.appendChild(root)

    info = doc.createElement('info')
    info_txt = doc.createTextNode('This file contains errors between predicted landmarks and groundtruth landmark on the testsets of ibugs and wflw')
    info.appendChild(info_txt)
    root.appendChild(info)

    images = doc.createElement('images')
    root.appendChild(images)

    for img_path in res_dict:
        image = doc.createElement('image')
        image.setAttribute('path', img_path)
        classes_data = data_dict[img_path][0]
        classes = doc.createElement('classes')
        classes.setAttribute('commonset', classes_data.attrib['commonset'])
        classes.setAttribute('challengeset', classes_data.attrib['challengeset'])
        classes.setAttribute('fullset', classes_data.attrib['fullset'])
        classes.setAttribute('wflw', classes_data.attrib['wflw'])
        classes.setAttribute('total', classes_data.attrib['total'])
        image.appendChild(classes)

        angle_data = data_dict[img_path][1]
        angle = doc.createElement('angle')
        angle.setAttribute('pitch', angle_data.attrib['pitch'])
        angle.setAttribute('yaw', angle_data.attrib['yaw'])
        angle.setAttribute('roll', angle_data.attrib['roll'])
        image.appendChild(angle)

        gt_pts = data_dict[img_path][2]
        res_pts = res_dict[img_path]
        error, norm_error = calc_error(gt_pts, res_pts, norm_type=norm_type)
        error_elem = doc.createElement('error')
        error_elem.setAttribute('pixel_error', str(error))
        error_elem.setAttribute('norm_error', str(norm_error))
        error_elem.setAttribute('norm_type', norm_type)
        image.appendChild(error_elem)
        images.appendChild(image)

    fp = open(xml_output, 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
    fp.close()

# def save_errors2(data_dict, res_dict, xml_output, norm_type='centers'):
#     root = ET.ElementTree()
#     first_elem= ET.Element('results')
#     info = ET.SubElement(first_elem, 'info')
#     info.txt = 'This file contains errors between predicted landmarks and groundtruth landmark on the testsets of ibugs and wflw'
#
#     images = ET.SubElement(first_elem, 'images')
#
#     for img_path in res_dict:
#         image = ET.SubElement(images, 'image')
#         image.setAttribute('path', img_path)
#         classes = data_dict[img_path][0]
#         image.appendChild(classes)
#         angle = data_dict[img_path][1]
#         image.appendChild(angle)
#
#         gt_pts = data_dict[img_path][2]
#         res_pts = res_dict[img_path]
#         error, norm_error = calc_error(gt_pts, res_pts, norm_type=norm_type)
#         error_elem = doc.createElement('error')
#         error_elem.setAttribute('pixel_error', str(error))
#         error_elem.setAttribute('norm_error', str(norm_error))
#         error_elem.setAttribute('norm_type', norm_type)
#         image.appendChild(error_elem)
#         images.appendChild(image)
#
#     fp = open(xml_output, 'w')
#     doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
#     fp.close()


def main(args):
    if args.gt_file=='' or args.res_file=='' or args.output_xml=='':
        print("groundtruth file, result file and output file should be specified!")
        exit(0)
    data_dict = read_xml(args.gt_file, pts_num=82)
    pred_dict = read_result(args.res_file, pts_num=82)
    save_errors(data_dict, pred_dict, args.output_xml, norm_type=args.norm_type)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', type=str, default='/home/slam/workspace/DL/alignment_method/align_untouch/data/test_data/untouch_testset.xml')
    parser.add_argument('--res_file', type=str, default='/home/slam/workspace/DL/untouch_projects/dms_methods/tmp_result/pts_path_untouch_testset_ljj.txt')
    parser.add_argument('--output_xml', type=str, default='/home/slam/workspace/DL/alignment_method/align_untouch/temp/untouch_testset_error_ljj.xml')
    parser.add_argument('--norm_type', type=str, default='centers')
    return parser.parse_args(argv)

if __name__ == '__main__':
    # img_dict = read_xml('/home/slam/workspace/DL/alignment_method/align_untouch/data/test_data/total_testset.xml')
    # for img_path in img_dict:
    #     print(img_path)
    #     print(img_dict[img_path][0])
    main(parse_arguments(sys.argv[1:]))

