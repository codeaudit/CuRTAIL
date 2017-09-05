from curtail import *
from model_architecture import this_model
from read_dataset import *
import scipy.io as sio



path_to_detector='mnist_detector'								#the detector model
path_to_baseline='mnist_baseline'								#the baseline victim model
path_to_images='Deepfool/5iterations/eps0.001/denoised4.mat'	#path to adversarial/denoised images, this path inherits the number of nonzeros for OMP
checkpoint_id=-4												#checkpoinint layer id
SP=0.01 															#security parameter for latent dictionary


#create and load victim model:
victim=this_model(X_train.shape[1:])
victim.load_weights(path_to_baseline)

sess=K.get_session()

sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
model=this_model(X_train.shape[1:])
model.compile(optimizer=sgd,
        loss=['categorical_crossentropy'],
        metrics=['accuracy'])

#create and load MRR model:
model_detector = Sequential()
model.summary()


for i in range(len(model.layers)):
	name=model.layers[i].name
	if name[0:4]!='drop':
		model_detector.add(model.layers[i])
	if model.layers[i].name==model.layers[checkpoint_id].name:
		checkpoint_id=i
		break


feature_shape=model_detector.output.get_shape()[1]
model_detector.add(l2_norm([n_output,feature_shape]))
model_detector.load_weights(path_to_detector)

#define gaussian pdf modules:
det=detector(model_detector,n_output)
det.fit_PCA_gaussians(X_train,y_train,var_abs=0.99)




det.set_pdf_thresholds(SP)


dic=sio.loadmat(path_to_images)

X_orig=dic['original_images']
X_reconst=dic['denoised_images']
y=np.ravel(dic['missed_to'])



victim_prediction=np.argmax(victim.predict(X_orig),axis=1)
victim_prediction_denoised=np.argmax(victim.predict(X_reconst),axis=1)
dict_suspicious=victim_prediction!=victim_prediction_denoised
detector_suspicious=det.investigate_samples(X_orig,y)
overall_suspicious=np.logical_or(dict_suspicious,detector_suspicious)

dict_sus_rate=np.sum(dict_suspicious)*1.0/len(dict_suspicious)*100
det_sus_rate=np.sum(detector_suspicious)*1.0/len(detector_suspicious)*100
over_sus_rate=np.sum(overall_suspicious)*1.0/len(overall_suspicious)*100
print 'input dictionary is suspected to %0.2f percent of images'%dict_sus_rate
print 'latent dictionary is suspected to %0.2f percent of images'%det_sus_rate
print 'overall, the modules are suspected to %0.2f percent of images'%over_sus_rate



