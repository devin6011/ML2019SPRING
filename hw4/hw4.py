import matplotlib
matplotlib.use('Agg')
import models
import numpy as np
np.random.seed(87)
import pandas as pd
import io
import keras
import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
tf.set_random_seed(87)
import keras.backend as K

print('Loading data...')
trainData = pd.read_csv(sys.argv[1], skiprows=[0], header=None)
X_train = pd.read_csv(io.StringIO(trainData.iloc[:, 1].to_csv(header=False, index=False)), sep='\s+', header=None).to_numpy(dtype=float)
y_train = trainData.iloc[:, 0].to_numpy(dtype=float)
print('Loading data done')

print('Preprocessing...')

X_train /= 255

y_train = keras.utils.to_categorical(y_train)
X_train = X_train.reshape((-1, 48, 48, 1))

print('Preprocessing done')

print('Loading models...')

model = models.Model4()
model.load_weights('model4.h5')

print('Loading models done')

print('Computing saliency...')

imgList = [2314, 1542, 27935, 28684, 18488, 8837, 22403]
X = X_train[imgList]
y = y_train[imgList]
#print(y)

session = K.get_session()

target = K.placeholder(shape=(None, 7))
loss = K.categorical_crossentropy(target, model.output)
grad = K.gradients(loss, model.input)

X_grad = session.run(grad, feed_dict={model.input: X, target: y})
saliency = np.abs(X_grad).squeeze()
X = X.squeeze()
for i in range(len(X)):
    #plt.imsave('fig0_' + str(i+1) + '.jpg', X[i], cmap='gray')
    plt.figure()
    plt.imshow(saliency[i], cmap='jet')
    plt.colorbar()
    plt.savefig(os.path.join(sys.argv[2], 'fig1_%d.jpg' % i))

print('Computing saliency done')

print('Visualizing filters...')
layer_dict = {layer.name: layer for layer in model.layers}
#print('Available layer names:')
#print(layer_dict.keys())

#layer_name = 'conv2d_2'
layer_names = list(filter(lambda x : 'conv' in x, layer_dict.keys()))
#layer_names = [layer_names[2], layer_names[4]]
layer_names = [layer_names[2]]
layer_nums = [30, 30, 30, 60, 60, 60, 90, 90, 90, 120, 120, 7]
#layer_nums = [layer_nums[2], layer_nums[4]]
layer_nums = [layer_nums[2]]
for layer_name, layer_num in zip(layer_names, layer_nums):
    if layer_num == 30:
        f, ax = plt.subplots(5, 6)
    elif layer_num == 60:
        f, ax = plt.subplots(6, 10)
    else:
        print('Error')
    for axis in ax.ravel():
        axis.axis('off')
    for filter_index in range(layer_num):
        print(filter_index)

        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, model.input)[0]
        grads /= K.sqrt(K.mean(K.square(grads))) + K.epsilon()
        iterate = K.function([model.input], [loss, grads])


        flag = True
        while flag:
            input_img_data = ((np.random.random((1, 48, 48, 1)) - 0.5) * 20 + 128) / 255

            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            lr = 0.001
            m = np.zeros((1, 48, 48, 1))
            v = np.zeros((1, 48, 48, 1))
            for i in range(100):
                loss_value, grads_value = iterate([input_img_data])
                m = beta1 * m + (1 - beta1) * grads_value
                v = beta2 * v + (1 - beta2) * grads_value ** 2
                mhat = m / (1 - beta1 ** (i+1))
                vhat = v / (1 - beta2 ** (i+1))
                input_img_data += lr * mhat / (np.sqrt(vhat) + epsilon)

                if loss_value <= K.epsilon():
                    break
            else:
                flag = False
        def postprocess_image(x):
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 0.1
            x += 0.5
            x = np.clip(x, 0, 1)
            x *= 255
            x = np.clip(x, 0, 255).astype('uint8')
            return x

        img = input_img_data[0]
        img = postprocess_image(img)

        im = ax.ravel()[filter_index].imshow(img.squeeze(), cmap='ocean')
        #plt.imsave('%s_filter_%d.png' % (layer_name, filter_index), img.squeeze())
    #plt.tight_layout()
    f.colorbar(im, ax=ax.ravel().tolist())
    #plt.savefig('fig2_1_%s.jpg' % (layer_name))
    plt.savefig(os.path.join(sys.argv[2], 'fig2_1.jpg'))

print('Visualizing filters done')

print('Visualizing filters output...')
img_num = 18857
X = X_train[img_num:img_num+1]

for layer_name, layer_num in zip(layer_names, layer_nums):
    if layer_num == 30:
        f, ax = plt.subplots(5, 6)
    elif layer_num == 60:
        f, ax = plt.subplots(6, 10)
    else:
        print('Error')
    for axis in ax.ravel():
        axis.axis('off')
    for filter_index in range(layer_num):
        print(filter_index)

        layer_output = layer_dict[layer_name].output
        filter_output = layer_output[:, :, :, filter_index]
        filterFunc = K.function([model.input], [filter_output])

        input_img_data = X

        input_img_data = filterFunc([input_img_data])

        def postprocess_image(x):
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 0.1
            x += 0.5
            x = np.clip(x, 0, 1)
            x *= 255
            x = np.clip(x, 0, 255).astype('uint8')
            return x

        img = input_img_data[0]
        img = postprocess_image(img)
        im = ax.ravel()[filter_index].imshow(img.squeeze(), cmap='ocean')
        #plt.imsave('%s_filter_%d_face.png' % (layer_name, filter_index), img.squeeze())
    f.colorbar(im, ax=ax.ravel().tolist())
    #plt.savefig('fig2_2_%s_face.jpg' % (layer_name))
    plt.savefig(os.path.join(sys.argv[2], 'fig2_2.jpg'))

print('Visualizing filters output done')

print('Lime-ing...')

from lime import lime_image
from skimage.segmentation import slic

X_train_rgb = np.concatenate([X_train, X_train, X_train], axis=3)

def predict(X):
    return model.predict(X[..., 0:1])

def segmentation(X):
    return slic(X)

for i, idx in enumerate(imgList):
    explainer = lime_image.LimeImageExplainer()
    np.random.seed(87)
    explanation = explainer.explain_instance(image=X_train_rgb[idx],
                                             classifier_fn=predict,
                                             segmentation_fn=segmentation
                                             )

    image, mask = explanation.get_image_and_mask(label=np.argmax(y_train[idx]),
                                                 positive_only=False,
                                                 hide_rest=False,
                                                 num_features=5,
                                                 min_weight=0.0
                                                 )

    plt.imsave(os.path.join(sys.argv[2], 'fig3_%d.jpg' % i), image, format='jpg')

print('Lime-ing done')
