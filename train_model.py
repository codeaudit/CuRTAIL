from curtail import *
from model_architecture import this_model
from read_dataset import *



#specify learning parameters:
batch_size = 100
epochs = 1
lr=0.0001
decay=1e-6
moment=0.9

#path to save the trained model:
path_to_save='mnist_baseline'

#define model to train
model=this_model(X_train.shape[1:])

sgd = SGD(lr=lr, decay=decay, momentum=moment, nesterov=False)


model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(X_train,Y_train,batch_size=batch_size,verbose=2,nb_epoch=epochs,validation_data=(X_test,Y_test))

model.save_weights(path_to_save)