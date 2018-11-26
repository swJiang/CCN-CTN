import os
import numpy as np

def data_load(train_dataType = 'notredame', test_dataType = 'yosemite'):

    dataDir = '/home/jsw/paper/triple_complex/%s_train_patch_pairs'%train_dataType
    dataFileName = '%s_500k_patch_pairs_image.npy'%train_dataType
    labelFileName = '%s_500k_patch_pairs_labels.npy'%train_dataType
    
    test_dataDir = '/home/jsw/paper/triple_complex/%s_test_patch_pairs'%test_dataType
    test_dataFileName = '%s_100k_patch_pairs_image.npy'%test_dataType
    test_labelFileName = '%s_100k_patch_pairs_labels.npy'%test_dataType
    
    
    dataPath = os.path.join(dataDir, dataFileName)
    train_images = np.load(dataPath)
    total_num = len(train_images)
    labelPath = os.path.join(dataDir, labelFileName)
    labels = np.load(labelPath)
    labels = np.reshape(labels, newshape=[total_num, 1])
    
    test_dataPath = os.path.join(test_dataDir, test_dataFileName)
    test_images = np.load(test_dataPath)
    test_total_num = len(test_images)
    test_labelPath = os.path.join(test_dataDir, test_labelFileName)
    test_labels = np.load(test_labelPath)
    test_labels = np.reshape(test_labels, newshape=[test_total_num, 1])
    
    return train_images,labels, test_images, test_labels


def triple_data_load(train_dataType = 'notredame', test_dataType = 'yosemite'):
    dataDir = '/home/jsw/paper/triple_complex/%s_triplet_patches_data' % train_dataType
    dataFileName = '%s_500k_triplet_patch_image.npy' % train_dataType
    test_dataType = 'liberty'
    test_dataDir = '/home/jsw/paper/triple_complex/%s_test_patch_pairs' % test_dataType
    test_dataFileName = '%s_100k_patch_pairs_image.npy' % test_dataType
    test_labelFileName = '%s_100k_patch_pairs_labels.npy' % test_dataType
    dataPath = os.path.join(dataDir, dataFileName)
    train_images = np.load(dataPath)
    test_dataPath = os.path.join(test_dataDir, test_dataFileName)
    test_images = np.load(test_dataPath)
    test_total_num = len(test_images)
    test_labelPath = os.path.join(test_dataDir, test_labelFileName)
    test_labels = np.load(test_labelPath)
    test_labels = np.reshape(test_labels, newshape=[test_total_num, 1])

    return train_images, test_images, test_labels