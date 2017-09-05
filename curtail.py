import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Lambda, Convolution2D, MaxPooling2D, Flatten,Dropout,Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
import pickle
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GM
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
from pylab import *
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
import matplotlib.image as mpimg
import pandas as pd
from operator import eq
from sklearn.metrics import confusion_matrix
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from progressbarsimple import ProgressBar

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	#    plt.text(j, i, cm[i, j],
	#             horizontalalignment = "center",
	#             color = "white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
def plt_all_clusters(projected,y,projected_adv=None):
	y=y.ravel()
	y=y.astype(np.int)
	classes=np.unique(y)
	n_class=classes.shape[0]
	colors=np.random.rand(n_class,3)
	cl=np.random.rand(0,3)
	classes_projected=np.random.rand(0,projected.shape[1])

	for i in classes:
		cl=np.concatenate((cl,np.repeat(colors[i:i+1],projected[y==i].shape[0],axis=0)))
		classes_projected=np.concatenate((classes_projected,projected[y==i]))



	area=np.repeat(np.pi*0.5,classes_projected.shape[0],axis=0)
	if projected_adv!=None:
		cl=np.concatenate((cl,np.repeat([[1,0,0]],projected_adv.shape[0],axis=0)))
		classes_projected=np.concatenate((classes_projected,projected_adv))
		area=np.concatenate((area,np.repeat(np.pi*5,projected_adv.shape[0],axis=0)))

	n=classes_projected.shape[0]
	inds=np.random.permutation(n)
	cl=cl[inds]
	area=area[inds]
	classes_projected=classes_projected[inds]
	if classes_projected.shape[1]==2:
		plt.scatter(classes_projected[:,0], classes_projected[:,1],s=area,c=cl,lw = 0)
	else:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(classes_projected[:,0], classes_projected[:,1], classes_projected[:,2], s=area, c=cl, lw = 0)

	#plt.show()
def print_tensor(t):
  print (t.op.name, t.get_shape().as_list())

class l2_norm (Layer):
	def __init__(self, centers_shape, **kwargs):
		with tf.variable_scope("cntr"):
			self.centers = tf.get_variable("centers",centers_shape)
		super(l2_norm, self).__init__(**kwargs)
	def build(self, input_shape):
		self.trainable_weights = [self.centers]
	def call(self, x, mask=None):
		return tf.nn.l2_normalize(x,1,1e-10)
	def get_output_shape_for(self,input_shape):
		return input_shape
	def set_centers(self,features,y_train,n_output,sess):
		centers=[]
		for i in range(n_output):
		    this_class = features[y_train.ravel()==i]
		    c = np.mean(this_class,axis=0)
		    centers.append(c)
		centers = np.asarray(centers)
		sess.run(self.centers.assign(centers))

def center_loss(label,features):
	"""Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
	(http://ydwen.github.io/papers/WenECCV16.pdf)
	adding additional loss to maximize the distance b/w cluster centers
	"""
	with tf.variable_scope("cntr",reuse=True):
		centers = tf.get_variable('centers')

	"""print label.get_shape()
				num_classes=10
				print num_classes
				costs=[]
				for i in range(num_classes):
					for j in range(i+1, num_classes):
						costs.append(1/tf.nn.l2_loss(centers[i]-centers[j]))
				fissher_coef=tf.add_n(costs)"""


	label=tf.to_int32(label)
	label=tf.argmax(label,axis=1)
	label = tf.reshape(label, [-1])
	centers_batch = tf.gather(centers, label)
	#coeffs_batch=tf.gather(fissher_coeffs, label)
	loss = tf.nn.l2_loss(features - centers_batch)
	return loss

class maximize_cent_dist(keras.callbacks.Callback):
	def __init__(self,sess,train_step,loss):
		self.sess=sess
		self.train_step=train_step
		self.loss=loss
	def on_batch_end(self, batch, logs={}):
		self.sess.run(self.train_step)
	def on_epoch_end(self, batch, logs={}):
		print 'center distance loss is',self.sess.run(self.loss)

class adversarial(object):
	def __init__(self,model,sess,batch_size,min_pix,max_pix,top_k,input_dtype):
		self.input_dtype=input_dtype
		self.sess=sess
		self.batch_size=batch_size
		self.model=model
		transposed_out=K.transpose(self.model.output)
		
		transposed_out_logit=K.transpose(self.model.layers[-2].get_output_at(-1))
		n_output=self.model.output.get_shape()[1]
		outputTensor = [transposed_out[i] for i in range(n_output)]
		outputTensor_logit=[transposed_out_logit[i] for i in range(n_output)]
		self.min_pix=min_pix
		self.max_pix=max_pix
		listOfVariableTensors = model.input
		zero_grad=tf.zeros(model.input.get_shape()[1:])
		ndims=model.input.get_shape().ndims
		self.y_=tf.placeholder(tf.float32,(self.batch_size,n_output))
		self.eps=tf.placeholder(tf.float32)
		self.found=tf.placeholder(tf.float32,shape=[batch_size])
		img_orig_shape=[batch_size]+self.model.input.get_shape().as_list()[1:]
		
		#define placeholders and variables for adversarial examples:
		im=np.zeros(img_orig_shape,dtype=np.float32)
		im_flat=im.reshape(batch_size,-1)
		initial_val=tf.constant(np.ones(im.reshape([-1,1]).shape),dtype=tf.float32)
		self.baseline_imgs_ph=tf.placeholder(tf.float32,img_orig_shape)
		self.perturbed_images=tf.Variable(im)
		self.pixel_not_changed_flat=tf.Variable(im.reshape([-1,1]))
		self.perturbed_images_flat=tf.Variable(im_flat)
		#ops that initialize input images for adversarial perturbation
		#1-used for FSG attack:
		self.img_baseline_init=self.perturbed_images.assign(self.baseline_imgs_ph)
		#2-used for JSMA attack:
		self.flat_img_baseline_init=self.perturbed_images_flat.assign(tf.reshape(self.baseline_imgs_ph,[self.batch_size,-1]))


		self.gradients =tf.convert_to_tensor([tf.gradients(outputTensor[i],listOfVariableTensors,gate_gradients=True) for i in range(n_output)])
		self.gradients_logits =tf.convert_to_tensor([tf.gradients(outputTensor_logit[i],listOfVariableTensors,gate_gradients=True) for i in range(n_output)])

		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.model.output), reduction_indices=[1]))
		self.cost_grad_wrt_input=tf.gradients(-cross_entropy,listOfVariableTensors,gate_gradients=True)
		self.cost_grad_wrt_input=tf.clip_by_value(self.cost_grad_wrt_input,-max_pix,max_pix)
		self.cost_grad_wrt_input=tf.squeeze(self.cost_grad_wrt_input,axis=[0])
		self.sign_cost_grad_wrt_input=max_pix*tf.sign(self.cost_grad_wrt_input)

		#fast sign gradient:
		self.FSG_pert=self.perturbed_images.assign(tf.clip_by_value(self.perturbed_images-self.eps*self.sign_cost_grad_wrt_input,min_pix,max_pix))
		
		#fast forward gradient (only positive gradients)
		self.indices_to_ph=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])
		indices_to = tf.convert_to_tensor([[self.indices_to_ph[i],0,i]for i in range(self.batch_size)])
		target_class_gradient2=tf.gather_nd(self.gradients,indices_to)
		sum_of_all=tf.reduce_sum(tf.reduce_sum(self.gradients,axis=0),axis=0)
		sum_of_all_except_target=sum_of_all-target_class_gradient2
		cond1=tf.to_float(sum_of_all_except_target<0)
		cond2=tf.to_float(target_class_gradient2>0)
		cond=cond1*cond2
		self.targeted_grads_forward=cond*target_class_gradient2*sum_of_all_except_target
		absol=tf.abs(self.targeted_grads_forward)
		mx=tf.reduce_max(absol,axis=[i for i in range(1,ndims)],keep_dims=True)
		self.targeted_grads_forward=self.targeted_grads_forward/mx*max_pix
		self.targeted_grads_forward=tf.clip_by_value(self.targeted_grads_forward,-max_pix,max_pix)
		self.init_pixel_not_changed=self.pixel_not_changed_flat.assign(initial_val)
		self.pixel_not_changed=tf.reshape(self.pixel_not_changed_flat,shape=img_orig_shape)
		self.saliency_map=tf.abs(self.targeted_grads_forward)*self.pixel_not_changed#cross-out pixels that have been changed
		sample_indx=np.zeros((batch_size,top_k))
		for i in range(batch_size):
			sample_indx[i,:]=i
		sample_indx=np.reshape(sample_indx,[-1,1])
		sample_indx=tf.constant(sample_indx,dtype=tf.int32)
		flat_sal=tf.reshape(self.saliency_map,[batch_size,-1])
		self.top_pixels_inds=tf.nn.top_k(flat_sal,k=top_k).indices
		self.top_pixels_inds_flat=tf.reshape(self.top_pixels_inds,[-1,1])
		self.top_pixels_inds_id=tf.concat([sample_indx,self.top_pixels_inds_flat],axis=1)
		self.updated_forward=tf.gather_nd(self.perturbed_images_flat,self.top_pixels_inds_id)
		self.saliency_not_zero=tf.sign(tf.gather_nd(flat_sal,self.top_pixels_inds_id))
		self.updated_forward=self.updated_forward+self.eps*max_pix*self.saliency_not_zero
		self.updated_forward=tf.clip_by_value(self.updated_forward,self.min_pix,self.max_pix)
		updated_zeros=tf.constant(np.zeros((batch_size*top_k)),dtype=tf.float32)
		self.JSMA_pert=[tf.scatter_nd_update(ref=self.perturbed_images_flat,indices=self.top_pixels_inds_id,updates=self.updated_forward)
		, tf.scatter_nd_update(ref=self.pixel_not_changed_flat,indices=self.top_pixels_inds_id,updates=updated_zeros)]
		self.perturbed_forward=tf.reshape(self.perturbed_images_flat,shape=img_orig_shape)

		#Deepfool:
		
		self.indices_from_ph=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])
		indices_from = tf.convert_to_tensor([[self.indices_from_ph[i],0,i]for i in range(self.batch_size)])
		from_class_gradient=tf.gather_nd(self.gradients_logits,indices_from)
		
		before_softmax=tf.transpose(self.model.layers[-2].get_output_at(-1))

		indices_from = tf.convert_to_tensor([[self.indices_from_ph[i],i]for i in range(self.batch_size)])
		from_class_logit=tf.gather_nd(before_softmax,indices_from)
		begin=np.zeros(len(before_softmax.get_shape().as_list()),dtype=np.int32)
		size=before_softmax.get_shape().as_list()
		size[1]=batch_size
		before_softmax=tf.slice(before_softmax,begin=begin,size=size)
		f_prime=before_softmax-from_class_logit
		perm=[1,0]+list(range(2,self.gradients_logits.get_shape().ndims))
		squeezed_grad=tf.transpose(self.gradients_logits,perm=perm)[0]
		size=squeezed_grad.get_shape().as_list()
		size[1]=batch_size
		begin=np.zeros(len(squeezed_grad.get_shape().as_list()),dtype=np.int32)
		squeezed_grad=tf.slice(squeezed_grad, begin=begin, size=size)
		w_prime=squeezed_grad-from_class_gradient
		#set zero values to one:(this is a way to avoid 0/0 devision for the true label class)
		self.f_prime_no_zero=tf.abs(f_prime)+1e-15
		shapes=w_prime.get_shape().as_list()
		w_prime_reshaped=tf.reshape(w_prime,(shapes[0],shapes[1],-1))
		self.w_prime_no_zero=tf.norm(w_prime_reshaped,axis=2)
		self.w_prime_no_zero=tf.clip_by_value(self.w_prime_no_zero,1e-20,1e20)
		criterion=self.f_prime_no_zero/self.w_prime_no_zero
		closest_class_id=tf.argmin(criterion,axis=0)
		closest_class_id=tf.reshape(closest_class_id,(-1,1))
		#print w_prime
		im_indices=tf.constant(np.asarray(range(batch_size)))
		im_indices=tf.reshape(im_indices,(-1,1))
		closest_class_id=tf.concat([closest_class_id,im_indices],axis=1)
		directions=tf.gather_nd(w_prime,closest_class_id)
		magnitudes=tf.gather_nd(criterion,closest_class_id)
		self.updates=tf.convert_to_tensor([directions[i]*magnitudes[i]*(1-self.found[i]) for i in range(batch_size)])
		self.Deepfool_pert=self.perturbed_images.assign(tf.clip_by_value(self.perturbed_images+self.eps*self.updates,min_pix,max_pix))

		

	def create_perturbed(self,X,grad_method,y,eps,max_iter,perturb_to=[None]):
		"""
		this function creates purturbed images from the baseline image using iterative gradients.
		baseline_img: the batch of images to be perturbed
		grad_method: method to generate and apply perturbation, can be one of the following:
			FSG: takes the gradient of categorical cross entropy, applies its sign as perturbation
			loss: takes the gradient of categorical cross entropy and applies it as the perturbation
			targeted: takes (gradient_of_label - gradiend_of_target) and applies it as the gradient.
			JSMA: uses the gradients defined here: "The Limitations of Deep Learning
								in Adversarial Settings, Equation.8"
		labels: true labels
		eps: epsilon for iterative perturbation
		max_iter: maximum iterations for perturbation
		perturb_to: the target class for the "targeted" gradient computation
		stop_criterion: criterion for stopping the iterative perturbation. default to None.
		"""
		#t0=time.time()
		shape= [0]+ list(X.shape[1:])
		missed_perturbed_images=np.array([]).reshape(shape).astype(self.input_dtype)
		missed_original_images=np.array([]).reshape(shape).astype(self.input_dtype)
		missed_from=np.array([], dtype=np.int64)
		missed_to=np.array([], dtype=np.int64)
		#myProgressBar=ProgressBar(nElements=50,nIterations=X.shape[0]/self.batch_size)
		for i in range(X.shape[0]/self.batch_size):
			#myProgressBar.progress(i)
			baseline_img=X[i*self.batch_size:(i+1)*self.batch_size]
			labels=y[i*self.batch_size:(i+1)*self.batch_size]
			pred=self.model.predict(x=baseline_img,batch_size=baseline_img.shape[0])
			prev_indx=np.argmax(pred,1)
			true_indx=np.argmax(labels,1)
			perturbed_example=baseline_img
			found=np.zeros((self.batch_size))

			if grad_method=='JSMA':
				target_label=perturb_to[i*self.batch_size:(i+1)*self.batch_size]
				self.sess.run(self.init_pixel_not_changed)
				self.sess.run(self.flat_img_baseline_init,feed_dict={self.baseline_imgs_ph:baseline_img})
			if grad_method=='FSG':
				self.sess.run(self.img_baseline_init,feed_dict={self.baseline_imgs_ph:baseline_img})
			if grad_method=='Deepfool':
				self.sess.run(self.img_baseline_init,feed_dict={self.baseline_imgs_ph:baseline_img})

			not_successfull=[i for i in range(perturbed_example.shape[0])]
			final_perturbed=np.zeros(perturbed_example.shape)
			missed_indices= []
			for j in range(max_iter):
				if grad_method=='FSG':
					self.sess.run(self.FSG_pert,feed_dict={self.model.input:perturbed_example, self.y_:labels,self.eps:eps})
					perturbed_example=self.sess.run(self.perturbed_images)
					pred=self.model.predict(x=perturbed_example.astype(self.input_dtype).astype(np.float32),batch_size=baseline_img.shape[0])
					indx=np.argmax(pred,1)
					found=(indx!=prev_indx)*(prev_indx==true_indx)
				if grad_method=='JSMA':
					self.sess.run(self.JSMA_pert,feed_dict={self.model.input:perturbed_example,self.indices_to_ph:target_label,self.eps:eps})
					perturbed_example=self.sess.run(self.perturbed_forward)
					pred=self.model.predict(x=perturbed_example.astype(self.input_dtype).astype(np.float32),batch_size=baseline_img.shape[0])
					indx=np.argmax(pred,1)
					found=(indx==target_label)*(indx!=prev_indx)*(prev_indx==true_indx)
				if grad_method=='Deepfool':
					self.sess.run(self.Deepfool_pert,feed_dict={self.model.input:perturbed_example,self.indices_from_ph:prev_indx,self.eps:eps,self.found:found})
					perturbed_example=self.sess.run(self.perturbed_images)
					pred=self.model.predict(x=perturbed_example.astype(self.input_dtype).astype(np.float32),batch_size=baseline_img.shape[0])
					indx=np.argmax(pred,1)
					found=(indx!=prev_indx)*(prev_indx==true_indx)
					#print self.sess.run(self.w_prime_no_zero,feed_dict={self.model.input:perturbed_example,self.indices_from_ph:prev_indx,self.eps:eps,self.found:found})

				for c in not_successfull:
					if found[c]:
						final_perturbed[c]=perturbed_example[c].astype(self.input_dtype)
						not_successfull.remove(c)
						missed_indices.append(c)
				if len(not_successfull)==0:
					#print 'early termination at iteration %d'%j
					break

			missed_indices=np.ravel(missed_indices)
			if len(missed_indices)>0:
				missed_perturbed_images=np.concatenate([missed_perturbed_images,final_perturbed[missed_indices]])
				missed_original_images=np.concatenate([missed_original_images,baseline_img[missed_indices]])
				l=labels[missed_indices]#model.predict(X_test[i*batch_size:(i+1)*batch_size],batch_size=batch_size)[missed_indices]
				l=np.argmax(l,axis=1)
				missed_from=np.concatenate([missed_from,l])
				l=self.model.predict(final_perturbed,batch_size=self.batch_size)[missed_indices]
				l=np.argmax(l,axis=1)
				missed_to=np.concatenate([missed_to,l])
				sm=np.sum(final_perturbed)
				if sm==nan:
					raise ValueError("created nan images somewhere")
			#print 'processed',(i+1)*batch_size,'images, found',missed_to.shape[0],'adversarial samples'
		#print 'found',missed_to.shape[0],'adversarial samples'
		return missed_original_images,missed_perturbed_images,missed_from,missed_to
	def compute_grad(self,X,labels):
		return self.sess.run(self.cost_grad_wrt_input,feed_dict={self.model.input:X, self.y_:labels})

class detector(object):
	#this class implements a detector that fits gaussian distributions over data samples
	def __init__(self, model, num_classes):
		self.model=model
		self.num_classes=num_classes
	def fit_PCA_gaussians(self,X_train=None,y_train=None,rank=None,var_abs=None,directory_generator=None,batch_size=100):
		"""
		This function fits Principal vectors to the features extracted from X_train, then
		fits Gaussians to the PCA-projected outputs of each class in the training data

		Either "rank" or "var_abs" (and not both) should be specified to determine the rank of the projection
		"rank" directly specifies the rank
		"var_abs" is the minimum accepted eigenvalue

		"""
		
		if directory_generator:
			features_train=np.zeros([0]+list(self.model.output.get_shape()[1:]))
			y_train=np.zeros((0))
			batch_size=directory_generator.batch_size
			num_images=directory_generator.samples
			print 'calculating features from %d training images'%num_images
			for index in range(num_images/batch_size):
				X,y=directory_generator.next()
				y=np.argmax(y,axis=1)
				p=self.model.predict(X,batch_size=batch_size)
				features_train=np.concatenate((features_train,p))
				y_train=np.concatenate((y_train,y))
			print features_train.shape,y_train.shape
		else:
			features_train=self.model.predict(X_train,batch_size=batch_size)

		#if not(hasattr(self, 'eigen_vecs')):
		print 'calculating PCA over matrix of dimensionality ',features_train.shape
		if rank!=None:
			self.rank=rank
			pca = PCA(n_components=rank)
			pca.fit(features_train)
			self.eigen_vecs=pca.components_
		if rank==None:
			pca = PCA()
			pca.fit(features_train)
			for i in range(1,pca.explained_variance_ratio_.shape[0]):
				var=np.sum(pca.explained_variance_ratio_[0:i])
				if var>=var_abs:
					self.eigen_vecs=pca.components_[0:i]
					self.rank=i
					print 'Setting the projection rank to %d to absorb %f of the total variance'%(self.rank,var_abs)
					break

		projected_train=np.dot(features_train,np.transpose(self.eigen_vecs))
		self.means_=[]
		self.covariances_=[]
		self.logpdf=[]
		print 'fitting Gaussian kernels to the projected features'

		for i in range(self.num_classes):
			this_class=projected_train[y_train.ravel()==i]
			gm=GM(n_components=1)
			gm.fit(this_class)
			mean=gm.means_[0].astype(np.float64)
			cov=gm.covariances_[0].astype(np.float64)
			this_class_pdf_hist=mvn(mean=mean,cov=cov).logpdf(this_class.astype(np.float64))
			self.logpdf.append(this_class_pdf_hist)
			#print np.min(this_class_pdf_hist),np.max(this_class_pdf_hist),np.var(this_class_pdf_hist)
			self.means_.append(gm.means_[0])
			self.covariances_.append(gm.covariances_[0])


			"""h,b=np.histogram(this_class_pdf_hist,bins=100)
			plt.subplot(3,4,i+1)
			plt.bar(b[0:-1],h,width=0.01)
			plt.xlabel('Value')
			plt.ylabel('Frequency')
		plt.show()"""
	def set_pdf_thresholds(self,SP):
		#"SP" is a security parameter between 0 and 100, higher SP ==> higher security and higher false positive
		self.thresholds=[1.5*np.percentile(self.logpdf[i],SP) for i in range(self.num_classes)]
	def investigate_samples(self,X,y):
		"""
		This function examines samples "X" that are predicted by the CNN as labels "y" to check for validity
		returns a vector of boolean values (suspicious) with each of its element determining if the correponding input is adversarial (True)
		"""
		y=y.ravel()

		features=self.model.predict(X)
		features=np.dot(features,np.transpose(self.eigen_vecs))

		suspicious=np.zeros((X.shape[0])).astype(np.bool)
		for i in range(self.num_classes):
			ind=np.where(y==i)
			to_examine=features[ind]
			prb=mvn(mean=self.means_[i],cov=self.covariances_[i]).logpdf(to_examine.astype(np.float64))
			sus=prb<self.thresholds[i]
			suspicious[ind]=sus
		return suspicious
	def extract_features(self,X):
		"""
		This function extracts features from inputs X, then projects them into PCA components
		"""
		features=self.model.predict(X)
		features=np.dot(features,np.transpose(self.eigen_vecs))
		return features
	def set_params(self,params):
		self.rank=params['rank']
		self.eigen_vecs=params['eigen_vecs']
		self.means_=params['means_']
		self.covariances_=params['covariances_']
		self.logpdf=params['logpdf']
class detector_dict(object):
	def __init__(self,num_classes,dic_size,patch_size,height,width,depth,transform_algorithm,sparsity_penalty=1):
		"""
		X: training data
		y: labels
		num_classes: number of categories ==> number of dictionaries
		dic_size: number of columns for each dictionary
		patch_size: size of each patch correspondng to a colomn of the dictionary
		sparsity penalty: maximum fraction of nonzeros in the reconstructed vectors
		"""
		self.num_classes=num_classes
		self.dic_size=dic_size
		self.patch_size=patch_size
		self.sparsity_penalty=sparsity_penalty
		self.transform_algorithm=transform_algorithm
		self.height=height
		self.width=width
		self.depth=depth
		self.DICT=[]
		self.dicos=[]
		self.DIFFFF=[]
	def learn_dictionaries(self,X,y):
		self.DICT=[]
		self.dicos=[]
		t00 = time()
		print 'learning dictionaries'
		myProgressBar = ProgressBar(nElements = 50, nIterations = self.num_classes)
		for class_id in range(self.num_classes):
			myProgressBar.progress(class_id)
			DIFF = []
			This_class_train = X[y.ravel()==class_id]
			#print('Extracting reference patches from training data...')
			t0 = time()
		
			buffer = []


			#print 'learnning dictionary for class', class_id

			for i in range(This_class_train.shape[0]):

				for j in range(self.depth):
					data = extract_patches_2d(This_class_train[i, :, :, j], self.patch_size, max_patches = 30)
					data = np.reshape(data, (len(data), -1))
					buffer.append(data)

			data = np.array(buffer).astype(np.float64)
			data = np.reshape(data, (data.shape[0]*data.shape[1],-1))
			data -= np.mean(data, axis=0)
			data /= np.std(data, axis=0)
			#print('Patch Extraction done in %.2fs.' % (time() - t0))
			#print('Learning the dictionary from refrence patches...')
			dico = MiniBatchDictionaryLearning(n_components=self.dic_size, alpha=self.sparsity_penalty, n_iter=200)
			dico.set_params(**self.transform_algorithm)
			V = dico.fit(data).components_
			self.dicos.append(dico)
			self.DICT.append(V)
		dt = time() - t00
		print('dictionary learning done in %.2fs.' % dt)
	def learn_dictionaries_generator(self,generator):
		num_batches=generator.samples/generator.batch_size
		self.DICT=[]
		self.dicos=[]
		buffers=[[] for i in range(self.num_classes)]
		t00 = time()
		print 'extracting patches'
		myProgressBar = ProgressBar(nElements = 50, nIterations = num_batches)
		for index in range(num_batches):
			myProgressBar.progress(index)
			X,y=generator.next()
			y=np.argmax(y,axis=1)
			for i in range(y.shape[0]):
				class_id=y[i]
				for j in range(self.depth):
					data = extract_patches_2d(X[i,:,:,j], self.patch_size, max_patches = 30)
					data = np.reshape(data, (len(data), -1))
					buffers[class_id].append(data)

		print 'learning dictionaries'
		myProgressBar = ProgressBar(nElements = 50, nIterations = self.num_classes)
		for class_id in range(self.num_classes):
			myProgressBar.progress(class_id)
			data = np.array(buffers[class_id])
			data=data.astype(np.float64)
			data = data.reshape(data.shape[0]*data.shape[1],-1)
			data -= np.mean(data, axis=0)
			data /= np.std(data, axis=0)
			#print('Patch Extraction done in %.2fs.' % (time() - t0))
			#print('Learning the dictionary from refrence patches...')
			dico = MiniBatchDictionaryLearning(n_components=self.dic_size, alpha=self.sparsity_penalty, n_iter=200)
			dico.set_params(**self.transform_algorithm)
			V = dico.fit(data).components_
			self.dicos.append(dico)
			self.DICT.append(V)
		dt = time() - t00
		print('dictionary learning done in %.2fs.' % dt)
	def denoise(self,im,y,downsample=None):
		"""
		This function uses OMP to denoise an input image, then returns the noise pattern
		"""
		dico=self.dicos[y]
		data = []
		for j in range(self.depth):
			data0 = extract_patches_2d(im[:, :, j], self.patch_size)
			data.append(np.reshape(data0, (len(data0), -1)))


		data = np.array(data).astype(np.float64)
		if downsample:
			target_shape=data.shape[1]
			rows=int(np.sqrt(target_shape))
			downsampled=data.reshape(data.shape[0],rows,rows,data.shape[2])
			downsampled=downsampled[:,::downsample,::downsample,:]
			"""for i in range(downsampled.shape[1]):
				for j in range(downsampled.shape[2]):
					im=downsampled[:,i,j,:].reshape(3,10,10).transpose(1,2,0).astype(np.uint8)
					plt.imshow(im)
					plt.show()"""
			d_h,d_w=downsampled.shape[1:3]
			data=downsampled.reshape(data.shape[0],downsampled.shape[1]*downsampled.shape[2],data.shape[-1])
		data = np.reshape(data, (data.shape[0]*data.shape[1],-1))
		intercept = np.mean(data, axis=0)
		data -= intercept


		reconstructions = np.zeros(im.shape)
		#dico.set_params(**self.transform_algorithm)
		code = dico.transform(data)
		patches = np.dot(code, dico.components_)
		patches += intercept
		patches = patches.reshape(len(data), *self.patch_size)
		if downsample:
			patches=patches.reshape(self.depth,d_h,d_w,patches.shape[-2],patches.shape[-1])
			patches=patches.repeat(downsample, axis=1).repeat(downsample, axis=2)

			patches=patches.reshape(patches.shape[0],patches.shape[1]*patches.shape[2],patches.shape[-2],patches.shape[-1])
			patches=patches.reshape(patches.shape[0]*patches.shape[1],patches.shape[-2],patches.shape[-1])


		for j in range(self.depth):
			reconstructions[:,:,j] = reconstruct_from_patches_2d(patches[j*patches.shape[0]//self.depth:(j+1)*patches.shape[0]//self.depth], (self.height, self.width))
		#plt.imshow(reconstructions)
		#plt.show()
		#difference = reconstructions.reshape(self.height * self.width * self.depth, 1) - im.reshape(self.height * self.width * self.depth, 1)

		return reconstructions

	def gather_noise_history(self,X,y):
		"""
			This function takes inputs "X" and labels "y", then computes the noises for each
			class and each example. The noises are saved in self.DIFFFF after calling this function
		"""
		self.DIFFFF=[]
		t0=time()
		for class_id in range(self.num_classes):
			DIFF=[]
			dico=self.dicos[class_id]
			This_class_train = X[y.ravel()==class_id]

			"""
			buffer=This_class_train.transpose((1,2,0,3))	#32,32,n_samples,n_channels
			#print buffer.shape
			buffer=buffer.reshape(buffer.shape[0],buffer.shape[1],-1)#32,32,n_samples*n_channels
			#print buffer.shape
			buffer=extract_patches_2d(buffer,self.patch_size)#625,8,8,n_samples*n_channels
			patch_per_im=buffer.shape[0]
			n_samples=This_class_train.shape[0]
			n_channels=This_class_train.shape[3]
			#print buffer.shape
			buffer=buffer.reshape(buffer.shape[0],buffer.shape[1],buffer.shape[2],This_class_train.shape[0],This_class_train.shape[3])#625,8,8,n_samples,n_channels
			buffer=buffer.transpose((4,0,1,2,3))#n_channels,625,8,8,n_samples
			buffer=buffer.reshape(buffer.shape[0]*buffer.shape[1],buffer.shape[2],buffer.shape[3],buffer.shape[4])#n_channels*625,8,8,n_samples
			intercept = np.mean(buffer, axis=0)#intercept has shape of: 8,8,n_samples
			#print 'intercept shape is:',intercept.shape
			buffer -= intercept
			buffer=buffer.transpose((3,0,1,2))#n_samples,n_channels*625,8,8
			buffer=buffer.reshape(buffer.shape[0]*buffer.shape[1],buffer.shape[2],buffer.shape[3])#n_samples*n_channels*625,8,8
			buffer=buffer.reshape(-1,buffer.shape[1]*buffer.shape[2])#n_samples*n_channels*625,8*8
			#print 'patched data shape is:',buffer.shape

			code = dico.transform(buffer)
			#print 'coded patches shape is:', code.shape

			patches = np.dot(code, self.DICT[class_id])#n_samples*n_channels*625,8*8
			patches=patches.reshape(n_samples,n_channels*patch_per_im,patch_size[0],patch_size[1])#n_samples,n_channels*625,8,8
			patches=patches.transpose((1,2,3,0))#n_channels*625,8,8,n_samples
			patches+=intercept
			patches=patches.reshape(n_channels,patch_per_im,patch_size[0],patch_size[1],n_samples)#n_channels,625,8,8,n_samples
			patches=patches.transpose((1,2,3,4,0))#625,8,8,n_samples,n_channels
			patches=patches.reshape(patch_per_im,patch_size[0],patch_size[1],n_samples*n_channels)#625,8,8,n_samples*n_channels

			reconstructed=reconstruct_from_patches_2d(patches=patches,image_size=(This_class_train.shape[1],This_class_train.shape[2],This_class_train.shape[0]*This_class_train.shape[3]))#32,32,n_samples*n_channels
			reconstructed=reconstructed.reshape(This_class_train.shape[1],This_class_train.shape[2],This_class_train.shape[0],This_class_train.shape[3])#32,32,n_samples,n_channels
			reconstructed=reconstructed.transpose((2,0,1,3))#n_samples,32,32,n_channels


			differences=(This_class_train-reconstructed)**2
			differences=differences.reshape(differences.shape[0],-1)
			differences=np.sum(differences,axis=1)
			"""


			print 'computing training noise for class',class_id
			myProgressBar = ProgressBar(nElements = 50, nIterations = This_class_train.shape[0])
			for i in range(This_class_train.shape[0]):
				myProgressBar.progress(i)
				im=self.denoise(This_class_train[i],class_id)
				noise=This_class_train[i]-im
				diff = np.sqrt(np.sum(im ** 2))
				DIFF.append(diff)

			self.DIFFFF.append(DIFF)
		t1=time()
		print 'total time sequential:%f'%(t1-t0)
	def set_ws_diffs(self,diffs):
		self.WS_DIFF=diffs
	def set_th_for_P(self,P,sample_from='all'):
		"""
		This function sets thresholds for each class based on percentile parameter "P"
		P is a number between 0 and 100
		"""
		if sample_from=='all':
			self.thresholds=[np.percentile(np.ravel(self.DIFFFF[i]),P) for i in range(self.num_classes)]
		if sample_from=='within_sphere':
			self.thresholds=[np.percentile(np.ravel(self.WS_DIFF[i]),P) for i in range(self.num_classes)]
		
	def set_noise_hist(self,DIFFFF):
		self.DIFFFF=DIFFFF
	def set_dics(self,dicos):

		self.dicos=dicos
		for i in range(self.num_classes):
			self.dicos[i].set_params(**self.transform_algorithm)
	def investigate_samples(self,X,y,noise=False):
		"""
		This function examines samples "X" that are predicted by the CNN as labels "y" to check for validity
		returns a vector of boolean values (suspicious) with each of its element determining if the correponding input is adversarial (True)
		"""
		y=y.ravel()
		suspicious=np.zeros((X.shape[0])).astype(np.bool)
		#print 'checking images'
		if noise:
			for i in range(X.shape[0]):
				diff = np.sqrt(np.sum(X[i] ** 2))
				suspicious[i]=diff>=self.thresholds[y[i]]
				#print diff,self.thresholds[y[i]]
		else:
			myProgressBar = ProgressBar(nElements = 50, nIterations = y.shape[0])
			for i in range(y.shape[0]):
				myProgressBar.progress(i)
				im=self.denoise(X[i],y[i])
				noise=X[i]-im
				diff = np.sqrt(np.sum(noise ** 2))
				suspicious[i]=diff>=self.thresholds[y[i]]
		return suspicious
	def denoise_batch(self,generator,downsample=None):
		images,labels=generator.next()
		labels=np.argmax(labels,axis=1)
		print 'denoising images ...'
		mpb=ProgressBar(nElements=50,nIterations=images.shape[0])
		denoised_im=np.zeros((0,self.width,self.height,self.depth))
		for i in range(images.shape[0]):
			mpb.progress(i)
			denoised_im=np.append(denoised_im,np.array([self.denoise(images[i],labels[i],downsample=downsample)]),axis=0)
		return denoised_im,images,labels
