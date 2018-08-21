from utilities import dataset
from utilities import preprocess

root_dir = '/home/public/nfs72/face/ibugs'
# output_hdf5 = '../tmpdata/output.hdf5'
output_tfrecords = './tmpdata/output.record'

dset = dataset.Dataset()
dset.get_datalist(root_dir, ['png', 'jpg'])
dset.save(output_tfrecords, format='tfrecords')
print('finished!')
