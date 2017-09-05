from curtail import *
from read_dataset import *
from model_architecture import *


path_to_save_detector='mnist_detector'
path_to_baseline='mnist_baseline'


#specify learning parameters:
batch_size = 100
epochs = 10
lr=0.001
decay=1e-6
moment=0.9
gamma=0.01 #hyper parameter defined in equation (11) of the paper

#specify the checkpoint layer id:
checkpoint_id=-4


sess=K.get_session()
model=this_model(X_train.shape[1:])
model.load_weights(path_to_baseline)



opt = SGD(lr=lr, decay=decay, momentum=moment, nesterov=False)
# Let's train the model using RMSprop
model.compile(optimizer=opt,
        loss=['categorical_crossentropy'],
        metrics=['accuracy'])


print model.evaluate(X_test,Y_test)
model_detector = Sequential()

for i in range(len(model.layers)):
	name=model.layers[i].name
	if name[0:4]!='drop':
		model_detector.add(model.layers[i])
	if model.layers[i].name==model.layers[checkpoint_id].name:
		checkpoint_id=i
		break
#get the shape of the checpointing layer:
feature_shape=model_detector.output.get_shape()[1]
Y_train_features=np_utils.to_categorical(y_train, feature_shape)
Y_test_features=np_utils.to_categorical(y_test, feature_shape)
#add a normalization layer after the checpointing layer
model_detector.add(l2_norm([n_output,feature_shape]))
checpoint_layer_id=len(model_detector.layers)-1

features = model_detector.predict(X_train, batch_size=batch_size)
model_detector.layers[-1].set_centers(features,y_train,n_output,sess)
#add the rest of the layers:
if drop:
	for i in range(checkpoint_id,len(model.layers)):
		name=model.layers[i].name
		if name[0:4]!='drop':
			model_detector.add(model.layers[i])



#model_detector.summary()


model_detector.compile(optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

model_detect_and_classify = Model(input=model_detector.input, output=[model_detector.output, model_detector.layers[checpoint_layer_id].output])


model_detect_and_classify.compile(optimizer=opt,
        loss=['categorical_crossentropy', center_loss],
        loss_weights=[1, gamma],
        metrics=['accuracy'])

model_detect_and_classify.fit(X_train, [Y_train,Y_train_features],
        batch_size=batch_size,
        nb_epoch=epochs,
        validation_data=(X_test,[Y_test,Y_test_features]),
        #callbacks=[cback],
        verbose=2
        )

model_detector=Model(model_detector.input,model_detector.layers[checpoint_layer_id].output)

model_detector.save_weights(path_to_save_detector)
