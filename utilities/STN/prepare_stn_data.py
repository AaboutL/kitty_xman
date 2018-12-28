import numpy as np
from utilities.data_preparation import utils
import cv2
from scipy import ndimage
import glob
from utilities.data_preparation.save_read_tfrecord import save_tfrecord, load_tfrecord

class Transform(object):
    def __init__(self, img_size, factor=0.2, initialization='rect', color=True):
        self.img_size = img_size
        self.factor = factor
        self.initialization = initialization
        self.color = color

    def collect_data(self, data_dirs, bbox_file, meanshape, start_id, img_num,is_mirror): # taking data from 0 to start_id as validation set
        img_paths = []
        landmarks = []
        bboxes = []
        if bbox_file is not None:
            bbox_dict = utils.load_bbox(bbox_file)
        for data_dir in data_dirs:
            img_paths_single = glob.glob(data_dir + "*.png")
            img_paths_single += glob.glob(data_dir + "*.jpg")

            for j in range(len(img_paths_single)):
                img_paths.append(img_paths_single[j])
                pts_path = img_paths_single[j][: -3] + 'pts'
                landmarks.append(utils.loadFromPts(pts_path))
                if bbox_file is not None:
                    img_path = img_paths_single[j]
                    bboxes.append(bbox_dict[img_path])
        img_paths = img_paths[start_id : start_id + img_num]
        landmarks = landmarks[start_id : start_id + img_num]
        bboxes = bboxes[start_id : start_id + img_num]
        print('imgpaths num: ', len(img_paths))

        mirror_list = [False for i in range(img_num)]
        if is_mirror:
            mirror_list = mirror_list + [True for i in range(img_num)]
            img_paths = np.concatenate((img_paths, img_paths))
            landmarks = np.vstack((landmarks, landmarks))
            bboxes = np.vstack((bboxes, bboxes))

        self.ori_landmarks = landmarks
        self.img_paths = img_paths
        self.mirror_list = mirror_list
        self.bboxes = bboxes
        self.meanshape = meanshape
        print('Collect date finished!')


    def load_image(self):
        self.imgs = []
        self.init_landmarks = []
        self.gtlandmarks = []
        print('data num: ', len(self.img_paths))

        for i in range(len(self.img_paths)):
            img = cv2.imread(self.img_paths[i])
            if self.color:
                if len(img.shape) == 2:
                    # cv2.merge([img, img, img], img)
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2:
                    # img = np.mean(img, axis=2)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.uint8)
            if self.mirror_list[i]:
                self.ori_landmarks[i] = utils.mirrorShape(self.ori_landmarks[i], img.shape)
                img = np.fliplr(img)
            if not self.color:
                img = img[:, :, np.newaxis]

            gtlandmark = self.ori_landmarks[i]
            if self.initialization == 'rect':
                bestFit = utils.bestFitRect(gtlandmark, self.meanshape)
            elif self.initialization == 'similarity':
                bestFit = utils.bestFit(gtlandmark, self.meanshape)
            elif self.initialization == 'bbox':
                bestFit = utils.bestFitRect(gtlandmark, self.meanshape, box=self.bboxes[i])

            self.imgs.append(img)
            self.init_landmarks.append(bestFit)
            self.gtlandmarks.append(gtlandmark)
        self.init_landmarks = np.array(self.init_landmarks)
        self.gtlandmarks = np.array(self.gtlandmarks)
        print('Load image finished!')

    def transform(self):
        out_imgs_cano = []
        out_gtlandmarks_cano = []

        out_imgs_skew = []
        out_gtlandmarks_skew = []
        Ts = []

        offset = np.array(self.img_size[::-1]) / 2
        scaled_shape_size = self.img_size[0] * (1 - self.factor)
        for i in range(len(self.imgs)):
            img = self.imgs[i]
            gt_landmark = self.gtlandmarks[i]
            bestFit = self.init_landmarks[i]

            gt_landmark_size = max(gt_landmark.max(axis=0) - gt_landmark.min(axis=0))
            scaled_gt_landmark = (gt_landmark - gt_landmark.mean(axis=0)) * scaled_shape_size / gt_landmark_size + gt_landmark.mean(axis=0)
            gt_landmark_skew = scaled_gt_landmark.copy() - scaled_gt_landmark.mean(axis=0)
            gt_landmark_skew += offset
            A_g, t_g = utils.bestFit(gt_landmark_skew, gt_landmark, True)
            A_g_inv = np.linalg.inv(A_g)
            t_g_inv = np.dot(-t_g, A_g_inv)

            gt_outimg_skew = np.zeros((self.img_size[0], self.img_size[1], img.shape[2]), dtype=np.uint8)
            for c in range(img.shape[2]):
                gt_outimg_skew[:,:, c] = ndimage.interpolation.affine_transform(img[:, :, c], A_g_inv, t_g_inv[[1, 0]], output_shape=self.img_size)


            bestFit_size = max(bestFit.max(axis=0) - bestFit.min(axis=0))
            scaled_shape = (bestFit - bestFit.mean(axis=0)) * scaled_shape_size / bestFit_size + bestFit.mean(axis=0)
            dest_shape = scaled_shape.copy() - scaled_shape.mean(axis=0)
            dest_shape += offset

            A, t = utils.bestFit(dest_shape, gt_landmark, True)
            gt_landmark_cano = np.dot(gt_landmark, A) + t

            delta_tx = np.asarray(self.img_size)/2 - (gt_landmark_cano.max(axis=0) - gt_landmark_cano.min(axis=0))/2 - gt_landmark_cano.min(axis=0)
            A_inv = np.linalg.inv(A)
            t_inv = np.dot(-(t + np.asarray(delta_tx)), A_inv)
            gt_landmark_cano = gt_landmark_cano + np.asarray(delta_tx)

            outimg = np.zeros((self.img_size[0], self.img_size[1], img.shape[2]), dtype=np.uint8)
            for c in range(img.shape[2]):
                outimg[:, :, c] = ndimage.interpolation.affine_transform(img[:, :, c], A_inv, t_inv[[1, 0]], output_shape=self.img_size)

            A_skew_cano, t_skew_cano = utils.bestFit(gt_landmark_cano, gt_landmark_skew, True)
            T = np.vstack([A_skew_cano, t_skew_cano])

            out_imgs_skew.append(gt_outimg_skew)
            out_gtlandmarks_skew.append(gt_landmark_skew)
            out_imgs_cano.append(outimg)
            out_gtlandmarks_cano.append(gt_landmark_cano)
            Ts.append(T)

            # for k in range(len(bestFit)):
            #     cv2.circle(outimg, (int(gt_landmark_cano[k][0]), int(gt_landmark_cano[k][1])), 2, (0, 0, 255), 2)
            #     cv2.circle(gt_outimg_skew, (int(gt_landmark_skew[k][0]), int(gt_landmark_skew[k][1])), 2, (0, 255, 0), 2)
            # cv2.imshow('gt', gt_outimg_skew)
            # cv2.imshow('out', outimg)
            # cv2.waitKey(0)
        self.out_imgs_skew = np.array(out_imgs_skew)
        self.out_imgs_cano = np.array(out_imgs_cano)
        self.out_gtlandmarks_skew = np.array(out_gtlandmarks_skew)
        self.out_gtlandmarks_cano = np.array(out_gtlandmarks_cano)
        self.Ts = np.array(Ts)
        print('Transform finished!')

    def save_tfrecord(self, output, params_num=6):
        self.Ts = np.reshape(self.Ts, (len(self.Ts), params_num)).astype(np.float32)
        self.out_imgs_skew = self.out_imgs_skew.astype(np.uint8)
        self.out_imgs_skew = np.squeeze(self.out_imgs_skew, axis=3)
        save_tfrecord(self.out_imgs_skew, self.Ts, output)
        print('Save tfrecord finished!')

def main():
    meanshape = np.load("../data/meanFaceShape.npz")["meanShape"]

    imageDirs = ["/home/slam/nfs132_0/landmark/dataset/ibugs/300W/01_Indoor/",
             "/home/slam/nfs132_0/landmark/dataset/ibugs/lfpw/trainset/",
             "/home/slam/nfs132_0/landmark/dataset/ibugs/helen/trainset/",
             "/home/slam/nfs132_0/landmark/dataset/ibugs/afw/"]
    output_train = '/home/slam/workspace/DL/alignment_method/align_untouch/temp/trans_train.record'
    output_validate = '/home/slam/workspace/DL/alignment_method/align_untouch/temp/trans_validate.record'

    trans_train = Transform([224, 224], color=False)
    trans_train.collect_data(imageDirs, None, meanshape, 100, 10000, True)
    # trans_train.collect_data(imageDirs, None, meanshape, 0, 2, True)
    trans_train.load_image()
    trans_train.transform()
    trans_train.save_tfrecord(output_train, 6)
    print('train set finished!')
    # exit(0)
    trans_train = Transform([224, 224], color=False)
    trans_train.collect_data(imageDirs, None, meanshape, 0, 100, False)
    trans_train.load_image()
    trans_train.transform()
    trans_train.save_tfrecord(output_validate, 6)
    print('validate set finished!')

'''
    img_path = '/home/slam/nfs132_0/landmark/dataset/ibugs/300W/01_Indoor/indoor_014.png'
    # img_path = '/home/slam/nfs132_0/landmark/dataset/ibugs/300W/01_Indoor/indoor_001.png'
    pts_path = '/home/slam/nfs132_0/landmark/dataset/ibugs/300W/01_Indoor/indoor_014.pts'
    img = cv2.imread(img_path)
    gt_landmark = np.genfromtxt(pts_path, delimiter=' ', skip_header=3, skip_footer=1)

    res_imgsize = [240, 240]
    scaled_img_size = 190
    bestFit = utils.bestFitRect(gt_landmark, meanshape) # put meanshape to the gt_landmark bbox
    bestFit_size = max(bestFit.max(axis=0) - bestFit.min(axis=0))
    scaled_shape = (bestFit - bestFit.mean(axis=0)) * scaled_img_size / bestFit_size + bestFit.mean(axis=0)
    dest_shape = scaled_shape.copy() - scaled_shape.mean(axis=0)
    offset = np.array(res_imgsize[::-1]) / 2
    dest_shape += offset

    dest_gt_landmark = gt_landmark.copy() - gt_landmark.mean(axis=0) + offset

    A, t = utils.bestFit(dest_shape, dest_gt_landmark, True)
    gt_landmark = np.dot(gt_landmark, A) + t
    delta_tx = res_imgsize[0]/2 - (gt_landmark.max(axis=0)[0] - gt_landmark.min(axis=0)[0])/2 - gt_landmark.min(axis=0)[0]
    A_inv = np.linalg.inv(A)
    t_inv = np.dot(-(t+np.asarray([delta_tx, 0])), A_inv)
    gt_landmark = gt_landmark + np.asarray([delta_tx, 0])

    outImg = np.zeros((res_imgsize[0], res_imgsize[1], 3), dtype=np.uint8)
    for i in range(3):
        outImg[:, :, i] = ndimage.interpolation.affine_transform(img[:, :, i], A_inv, t_inv[[1, 0]], output_shape=res_imgsize)
        # outImg[:, :, i] = ndimage.interpolation.affine_transform(img[:, :, i], A, t[[1, 0]], output_shape=res_imgsize)
    cv2.imshow('out', outImg)

    for i in range(len(bestFit)):
        cv2.circle(img, (int(bestFit[i][0]), int(bestFit[i][1])), 2, (0, 255, 0), 2)
        cv2.circle(outImg, (int(gt_landmark[i][0]), int(gt_landmark[i][1])), 2, (0, 0, 255), 2)
        cv2.circle(outImg, (int(dest_gt_landmark[i][0]), int(dest_gt_landmark[i][1])), 2, (0, 255, 0), 2)
        cv2.circle(img, (int(dest_shape[i][0]), int(dest_shape[i][1])), 2, (255, 0, 0), 2)
    cv2.imshow('a', img)
    cv2.imshow('out', outImg)
    cv2.waitKey(0)
'''
if __name__ == '__main__':
    main()
