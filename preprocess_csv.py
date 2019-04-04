import numpy as np
import pandas as pd


if __name__ == '__main__':
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
