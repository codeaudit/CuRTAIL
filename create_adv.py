from curtail import *
from read_dataset import *
from model_architecture import *
import scipy.io as sio

"""
generate adversarial samples and save them. a dictionary object with folowing items is saved:
'original_images': clean images
'perturbed_images': adversarial samples 
'missed_from': predictions on the original images
'missed_to': predictions on the adversarial samples
"""

#specify the file containing the victim model
victim_model_path='mnist_baseline'
#specify attack alforithm. Possible values are FSG, JSMA, and Deepfool:
method='Deepfool'
#specify number of iterations for the attack algorithm:
niters=[5]
#specify different values for parameter epsilon to explore:
epsilons=[0.001]
#batch_size to process samples:
batch_size=100
#path to save adversarial samples
path_to_save_samples='.'



#construct victim model:
model=this_model(input_shape=X_train.shape[1:])
#reload the weights of the pre-trained model:
model.load_weights(victim_model_path)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


#remove dropout layers:
model1=Sequential()
for i in range(len(model.layers)):
	name=model.layers[i].name
	if name[0:4]!='drop':
		model1.add(model.layers[i])
model=model1

#pick the one-to-last prediction for each input (used as the target class for JSMA attack):
pred=model.predict(X_test)
p=np.argsort(pred,axis=1)
p=p[:,-2]


sess = K.get_session()

adv=adversarial(model,sess,batch_size,min_pix=0,max_pix=1,top_k=1,input_dtype=np.float32)
#sess.graph.finalize()

for max_iter in niters:
	if method=='FSG':
		max_iter=1
	for eps in epsilons:
		missed_original_images,missed_perturbed_images,missed_from,missed_to=adv.create_perturbed(X_test,method,Y_test,eps,max_iter,p)
		if missed_original_images.shape[0]>0:
			print 'saving', missed_original_images.shape[0],'images for epsilon =',eps
			dic={'original_images':missed_original_images,'perturbed_images':missed_perturbed_images,'missed_from':missed_from,'missed_to':missed_to}
			method_dir=os.path.join(path_to_save_samples,method)
			if(os.path.exists(method_dir)==False):
				os.mkdir(method_dir)
			iter_dir=os.path.join(method_dir,str(max_iter)+'iterations')
			if(os.path.exists(iter_dir)==False):
				os.mkdir(iter_dir)
			save_dir=os.path.join(iter_dir,'eps'+str(eps))
			if not(os.path.exists(save_dir)):
				os.mkdir(save_dir)
			pydir=os.path.join(save_dir,'perturbed.pkl')
			matdir=os.path.join(save_dir,'perturbed.mat')
			foo=open(pydir,'wb')
			pickle.dump(dic,foo)
			foo.close()
			sio.savemat(matdir,dic)
		else:
			print 'could not craft any adversarial samples for eps = %f niters=%d'%(eps,max_iter)
	if method=='FSG':
		break;