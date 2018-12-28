import glob
import os

import cv2
import numpy as np
from scipy import ndimage

from utilities.data_preparation import utils
from utilities.data_preparation.save_read_tfrecord import save_tfrecord, load_tfrecord


class DataServer(object):
    def __init__(self, imgsize=[112, 112], frameFraction=0.25, initialization='rect', color=False):
        self.imgsize = imgsize
        self.frameFraction = frameFraction
        self.initialization = initialization
        self.meanshape = np.array([])
        self.imgpaths = []
        self.ori_landmarks = []
        self.color = color


    def save_data(self, save_folder):
        for i in range(len(self.imgs)):
            name = '{:0>6}'.format(str(i))
            img_path = os.path.join(save_folder, name+'.jpg')
            pts_path = os.path.join(save_folder, name+'.pts')
            print('img_path %s' % img_path)
            print('pts_path %s' % pts_path)
            # for j in range(len(self.gtlandmarks[i])):
            #     cv2.circle(self.imgs[i], (int(self.gtlandmarks[i][j][0]), int(self.gtlandmarks[i][j][1])), 2, (0, 255, 0), 2)
            # cv2.imshow("i", self.imgs[i])
            # cv2.waitKey(0)
            cv2.imwrite(img_path, self.imgs[i])
            utils.saveToPts(pts_path, self.gtlandmarks[i])

    def read_tfrecord(self, input, pts_num=68, img_shape=[112,112], is_shuffle=True):
        load_tfrecord(input, pts_num=pts_num, img_shape=img_shape, is_shuffle=is_shuffle)

    def save_tfrecord(self, output, pts_num=68):
        self.gtlandmarks = np.reshape(self.gtlandmarks, (len(self.gtlandmarks), pts_num*2)).astype(np.float32)
        self.imgs = self.imgs.astype(np.uint8)
        self.imgs = np.squeeze(self.imgs,axis=3)
        print("img shape", np.shape(self.imgs))
        print("img type: ", self.imgs[0].dtype)
        print("pts shape", np.shape(self.gtlandmarks))
        print("pts type: ", self.gtlandmarks[0].dtype)
        save_tfrecord(self.imgs, self.gtlandmarks, output)

    def collect_data(self, data_dirs, bbox_file, meanshape, start_id, img_num, is_mirror): # taking data from 0 to start_id as validation set
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


    def load_image(self):
        self.imgs = []
        self.init_landmarks = []
        self.gtlandmarks = []

        for i in range(len(self.img_paths)):
            img = cv2.imread(self.img_paths[i])
            if self.color:
                if len(img.shape) == 2:
                    # cv2.merge([img, img, img], img)
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2:
                    img = np.mean(img, axis=2)
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

    def gen_perturbations(self, num_perturbation, perturbations):
        self.perturbations = perturbations
        meanshape_size = max(self.meanshape.max(axis=0) - self.meanshape.min(axis=0))
        destshape_size = min(self.imgsize) * (1 - 2*self.frameFraction)
        scaled_meanshape = self.meanshape * destshape_size / meanshape_size


        new_imgs = []
        new_gtlandmarks = []
        new_initlandmarks = []

        translationMultX, translationMultY, rotationStdDev, scaleStdDev = perturbations
        rotationStdDevRad = rotationStdDev * np.pi / 180
        translationStdDevX = translationMultX * (scaled_meanshape[:, 0].max() - scaled_meanshape[:, 0].min())
        translationStdDevY = translationMultY * (scaled_meanshape[:, 1].max() - scaled_meanshape[:, 1].min())

        for i in range(self.init_landmarks.shape[0]):
            for j in range(num_perturbation):
                tmp_init = self.init_landmarks[i].copy()
                angle = np.random.normal(0, rotationStdDevRad)
                offset = [np.random.normal(0, translationStdDevX), np.random.normal(0, translationStdDevY)]
                # scaling = np.random.normal(1, scaleStdDev)
                scaling = 1
                R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

                # translate, scale, rotate
                tempInit = tmp_init + offset
                tempInit = (tempInit - tempInit.mean(axis=0)) * scaling + tempInit.mean(axis=0)
                tempInit = np.dot(R, (tempInit - tempInit.mean(axis=0)).T).T + tempInit.mean(axis=0) # must subtract mean first

                # for k in range(len(tempInit)):
                #     cv2.circle(self.imgs[i].squeeze(), (int(tempInit[k][0]), int(tempInit[k][1])), 2, (0, 255, 0), 2)
                # cv2.imshow('img', self.imgs[i].squeeze())

                tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], tempInit, self.gtlandmarks[i])
                # print(self.imgs[i].shape)
                # print("tmp: ", tempImg.shape)
                # for k in range(len(tempGroundTruth)):
                    # cv2.circle(tempImg, (int(tempGroundTruth[k][0]), int(tempGroundTruth[k][1])), 2, (0, 255, 0))
                    # cv2.circle(tempImg, (int(tempInit[k][0]), int(tempInit[k][1])), 2, (0, 255, 0))
                # cv2.imshow("i", tempImg)
                # cv2.imshow('img', self.imgs[i].squeeze())
                # cv2.waitKey(0)

                # new_imgs.append(tempImg.transpose((1, 2, 0)))
                # print(tempImg.shape)
                new_imgs.append(tempImg)
                new_initlandmarks.append(tempInit)
                new_gtlandmarks.append(tempGroundTruth)
        self.imgs = np.array(new_imgs)
        self.initLandmarks = np.array(new_initlandmarks)
        self.gtlandmarks = np.array(new_gtlandmarks)

    def CropResizeRotate(self, img, initShape, groundTruth):
        # initShape is random perturbated scaled meanshape
        meanShapeSize = max(self.meanshape.max(axis=0) - self.meanshape.min(axis=0))
        # here scale the shape size
        # destShapeSize = min(self.imgsize) * (1 - 2 * self.frameFraction)
        destShapeSize = min(self.imgsize) * 0.9

        scaledMeanShape = self.meanshape * destShapeSize / meanShapeSize

        destShape = scaledMeanShape.copy() - scaledMeanShape.mean(axis=0)
        offset = np.array(self.imgsize[::-1]) / 2
        destShape += offset

        # tmpdst = np.zeros((self.imgsize[0], self.imgsize[1], 1), dtype=np.uint8)
        # tmpini = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        # for m in range(len(destShape)):
        #     cv2.circle(tmpdst, (int(destShape[m][0]), int(destShape[m][1])), 1, 255)
        #     cv2.circle(tmpini, (int(initShape[m][0]), int(initShape[m][1])), 2, 125)
        # cv2.imshow("tmpdst", tmpdst)
        # cv2.imshow("tmpini", tmpini)
        # cv2.waitKey(0)
        A, t = utils.bestFit(destShape, initShape, True)

        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)
        # T = np.hstack((A2, t2[[1, 0]].reshape(2, 1)))
        outImg = np.zeros((self.imgsize[0], self.imgsize[1], img.shape[2]), dtype=img.dtype)
        for i in range(img.shape[2]):
            outImg[:, :, i] = ndimage.interpolation.affine_transform(img[:, :, i], A2, t2[[1, 0]], output_shape=self.imgsize)
        # cv2.imshow("out", outImg)
        # cv2.waitKey(0)
        # outImg = cv2.warpAffine(img, T, (self.imgsize[0], self.imgsize[1]))

        initShape = np.dot(initShape, A) + t

        groundTruth = np.dot(groundTruth, A) + t
        return outImg, initShape, groundTruth

    def NormalizeImages(self, imageServer=None):
        self.imgs = self.imgs.astype(np.float32)
        if imageServer is None:
            self.meanImg = np.mean(self.imgs, axis=0)
        else:
            self.meanImg = imageServer.meanImg
        self.imgs = self.imgs - self.meanImg

        if imageServer is None:
            self.stdDevImg = np.std(self.imgs, axis=0)
        else:
            self.stdDevImg = imageServer.stdDevImg
        self.imgs = self.imgs / self.stdDevImg

        meanImg = self.meanImg - self.meanImg.min()
        meanImg = 255 * meanImg / meanImg.max()
        meanImg = meanImg.astype(np.uint8)
        if self.color:
            cv2.imwrite('data/meanImg.jpg', meanImg)
        else:
            print(meanImg.shape)
            cv2.imwrite('data/meanImg.jpg', meanImg.squeeze())

        stdDevImg = self.stdDevImg - self.stdDevImg.min()
        stdDevImg = 255 * stdDevImg / stdDevImg.max()
        stdDevImg = stdDevImg.astype(np.uint8)
        if self.color:
            cv2.imwrite('data/stdDevImg.jpg', stdDevImg)
        else:
            cv2.imwrite('data/stdDevImg.jpg', stdDevImg.squeeze())

