import numpy as np
import pandas as pd
import os


def process_fcn_label_merge():
    path_ = 'E:/Data/ShipDetection/FCN/train.csv'
    masks = pd.read_csv(path_)
    masks['multi'] = masks.duplicated(['ImageId'], False)
    reserve = masks[masks['multi'].isin([False])]
    process = masks[masks['multi'].isin([True])]
    process = process.groupby('ImageId')
    process = process.apply(lambda x: ' '.join(x['EncodedPixels'].to_list()))
    process = process.reset_index()
    process['EncodedPixels'] = process[0]
    process.drop(0)
    result = pd.concat([process, reserve], ignore_index=True)
    result.to_csv('E:/Data/ShipDetection/FCN/train2.csv')


def bin_to_csv():
    path_ = 'E:/Data/ShipDetection/CNN/'
    neg = os.listdir(path_ + 'negative')
    pos = os.listdir(path_ + 'ship')
    columns = ['label']
    neg = pd.DataFrame.from_dict({i: 0. for i in neg}, orient='index', columns=columns)
    pos = pd.DataFrame.from_dict({i: 1. for i in pos}, orient='index', columns=columns)
    save = pos.append(neg)
    save = save.reset_index()
    save.to_csv(path_ + 'labels.csv', index=False)


if __name__ == '__main__':
    bin_to_csv()
