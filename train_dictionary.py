from curtail import *
from read_dataset import *
path_to_save_dictionaries='mnist_dicts.mat'
#specify the dimensions of images in the datase:
height=28
width=28
depth=1
#specify patch_size for dictionary elements:
patch_size=[7,7]

#reshape images to original shape:
X_train=X_train.reshape(-1,height,width,depth)
X_test=X_test.reshape(-1,height,width,depth)

#define parameters for transform algorithm:
transform_algorithm = {'transform_algorithm':'omp','transform_n_nonzero_coefs': 5}
dict_det=detector_dict(n_output,dic_size=225,patch_size=patch_size,height=height,width=width,depth=depth,transform_algorithm=transform_algorithm,sparsity_penalty=1)

#learn dictionaries:
dict_det.learn_dictionaries(X_train,y_train)

dictionaries=[]

for dic in dict_det.dicos:
	d=dic.components_
	d=np.transpose(d) 	#the matlab omp toolbox works with the transpose of this dictionary
	dictionaries.append(d)
dictionaries=np.asarray(dictionaries)

#save dictionaries in matlab format:
dictionaries={'class_dictionaries':dictionaries}
import scipy.io as sio
sio.savemat(path_to_save_dictionaries,dictionaries)