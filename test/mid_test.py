from utilities import dataset
from utilities import preprocess

root_dir = '/home/public/nfs72/face/ibugs'
output_img = '/home/public/nfs132_1/hanfy/align/ibugs/images.txt'
output_pts = '/home/public/nfs132_1/hanfy/align/ibugs/points.txt'
output_hdf5 = '../tmpdata/output.hdf5'

dset = dataset.Dataset(root_dir)
dset.get_datalist(root_dir, ['png', 'jpg'])
dset.save(output_hdf5)
print('finished!')
