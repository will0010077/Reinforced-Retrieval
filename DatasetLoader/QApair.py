import sys
sys.path.append('../..')
import DatasetLoader.dataset as dataset
import h5py

if __name__ == '__main__':

    qadataset = dataset.QAPairDataset(num_samples=100)
    for x in qadataset:
        print(x)

    #demo
    #print(inspect_h5py_file('app/data/segmented_data.h5'))
