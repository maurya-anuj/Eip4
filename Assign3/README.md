Assignment 3

# Define the model
model = Sequential()

model.add(SeparableConv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))  # 32, 3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same'))  # 32, 5
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same'))  # 32, 7
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, (3, 3), padding='same'))  # 32, 9
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, (3, 3), padding='same'))  # 32, 11
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))  # 16, 12
model.add(Dropout(0.1))

model.add(Convolution2D(32, 1, activation='relu', use_bias=False))  # 16, 12
model.add(BatchNormalization())

model.add(SeparableConv2D(32, (3, 3), padding='same'))  # 16, 16
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same'))  # 16, 20
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same'))  # 16, 24
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same'))  # 16, 28
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, (3, 3), padding='same'))  # 16, 32
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))  # 8, 34
model.add(Dropout(0.1))

model.add(Convolution2D(32, 1, activation='relu', use_bias=False))  # 8, 34
model.add(BatchNormalization())

model.add(SeparableConv2D(32, (3, 3), padding='same'))  # 8, 42
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same'))  # 8, 50
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same'))  # 8, 58
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3)))  # 6, 66
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, (3, 3)))  # 4, 74
model.add(Activation('relu'))
model.add(BatchNormalization())


model.add(Convolution2D(10, 1, use_bias=False)) # 4, 74
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))



Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.2.
390/390 [==============================] - 63s 163ms/step - loss: 1.8290 - acc: 0.3102 - val_loss: 13.1181 - val_acc: 0.1067
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.1516300227.
390/390 [==============================] - 53s 137ms/step - loss: 1.4631 - acc: 0.4585 - val_loss: 10.1724 - val_acc: 0.1657
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.1221001221.
390/390 [==============================] - 53s 136ms/step - loss: 1.2184 - acc: 0.5550 - val_loss: 3.9804 - val_acc: 0.3757
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.1021972407.
390/390 [==============================] - 53s 137ms/step - loss: 1.0492 - acc: 0.6209 - val_loss: 3.6731 - val_acc: 0.4171
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0878734622.
390/390 [==============================] - 53s 136ms/step - loss: 0.9346 - acc: 0.6649 - val_loss: 3.9292 - val_acc: 0.3626
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0770712909.
390/390 [==============================] - 53s 137ms/step - loss: 0.8484 - acc: 0.6977 - val_loss: 2.0824 - val_acc: 0.5181
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0686341798.
390/390 [==============================] - 53s 137ms/step - loss: 0.7869 - acc: 0.7226 - val_loss: 1.1191 - val_acc: 0.6506
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0618620476.
390/390 [==============================] - 53s 136ms/step - loss: 0.7359 - acc: 0.7391 - val_loss: 1.2353 - val_acc: 0.6433
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0563063063.
390/390 [==============================] - 53s 137ms/step - loss: 0.6952 - acc: 0.7565 - val_loss: 1.3714 - val_acc: 0.6086
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0516662361.
390/390 [==============================] - 53s 136ms/step - loss: 0.6597 - acc: 0.7685 - val_loss: 0.8652 - val_acc: 0.7112
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0477326969.
390/390 [==============================] - 53s 136ms/step - loss: 0.6297 - acc: 0.7802 - val_loss: 1.1866 - val_acc: 0.6223
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.044355733.
390/390 [==============================] - 53s 136ms/step - loss: 0.6061 - acc: 0.7886 - val_loss: 0.7029 - val_acc: 0.7629
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0414250207.
390/390 [==============================] - 53s 136ms/step - loss: 0.5816 - acc: 0.7956 - val_loss: 0.7316 - val_acc: 0.7571
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0388575869.
390/390 [==============================] - 53s 136ms/step - loss: 0.5581 - acc: 0.8048 - val_loss: 0.8944 - val_acc: 0.7107
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.036589828.
390/390 [==============================] - 53s 136ms/step - loss: 0.5452 - acc: 0.8118 - val_loss: 0.6629 - val_acc: 0.7762
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0345721694.
390/390 [==============================] - 53s 136ms/step - loss: 0.5187 - acc: 0.8190 - val_loss: 0.6743 - val_acc: 0.7771
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.0327653997.
390/390 [==============================] - 53s 136ms/step - loss: 0.5080 - acc: 0.8231 - val_loss: 0.8296 - val_acc: 0.7363
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0311380975.
390/390 [==============================] - 53s 136ms/step - loss: 0.4899 - acc: 0.8276 - val_loss: 0.8762 - val_acc: 0.7229
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0296647879.
390/390 [==============================] - 53s 136ms/step - loss: 0.4748 - acc: 0.8351 - val_loss: 0.5681 - val_acc: 0.8044
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0283245999.
390/390 [==============================] - 53s 136ms/step - loss: 0.4610 - acc: 0.8388 - val_loss: 0.6387 - val_acc: 0.7890
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.027100271.
390/390 [==============================] - 53s 136ms/step - loss: 0.4532 - acc: 0.8420 - val_loss: 0.7339 - val_acc: 0.7663
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0259773997.
390/390 [==============================] - 53s 136ms/step - loss: 0.4402 - acc: 0.8454 - val_loss: 0.6122 - val_acc: 0.7986
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0249438763.
390/390 [==============================] - 53s 136ms/step - loss: 0.4292 - acc: 0.8497 - val_loss: 0.6461 - val_acc: 0.7884
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0239894446.
390/390 [==============================] - 53s 136ms/step - loss: 0.4194 - acc: 0.8535 - val_loss: 0.5178 - val_acc: 0.8248
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0231053604.
390/390 [==============================] - 53s 136ms/step - loss: 0.4096 - acc: 0.8577 - val_loss: 0.6051 - val_acc: 0.7964
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0222841226.
390/390 [==============================] - 53s 136ms/step - loss: 0.3986 - acc: 0.8590 - val_loss: 0.6813 - val_acc: 0.7848
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0215192597.
390/390 [==============================] - 53s 136ms/step - loss: 0.3933 - acc: 0.8620 - val_loss: 0.5566 - val_acc: 0.8207
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0208051597.
390/390 [==============================] - 53s 136ms/step - loss: 0.3823 - acc: 0.8662 - val_loss: 0.5545 - val_acc: 0.8204
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0201369311.
390/390 [==============================] - 53s 136ms/step - loss: 0.3771 - acc: 0.8691 - val_loss: 0.5648 - val_acc: 0.8178
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0195102917.
390/390 [==============================] - 53s 136ms/step - loss: 0.3694 - acc: 0.8708 - val_loss: 0.5254 - val_acc: 0.8257
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0189214759.
390/390 [==============================] - 53s 136ms/step - loss: 0.3640 - acc: 0.8734 - val_loss: 0.4943 - val_acc: 0.8367
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0183671595.
390/390 [==============================] - 53s 136ms/step - loss: 0.3552 - acc: 0.8751 - val_loss: 0.4707 - val_acc: 0.8450
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0178443969.
390/390 [==============================] - 53s 137ms/step - loss: 0.3482 - acc: 0.8780 - val_loss: 0.4643 - val_acc: 0.8476
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0173505682.
390/390 [==============================] - 53s 136ms/step - loss: 0.3416 - acc: 0.8796 - val_loss: 0.4697 - val_acc: 0.8440
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0168833361.
390/390 [==============================] - 53s 137ms/step - loss: 0.3375 - acc: 0.8825 - val_loss: 0.4830 - val_acc: 0.8427
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0164406083.
390/390 [==============================] - 53s 136ms/step - loss: 0.3318 - acc: 0.8832 - val_loss: 0.4861 - val_acc: 0.8396
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0160205062.
390/390 [==============================] - 53s 136ms/step - loss: 0.3267 - acc: 0.8853 - val_loss: 0.5739 - val_acc: 0.8177
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0156213387.
390/390 [==============================] - 53s 136ms/step - loss: 0.3235 - acc: 0.8866 - val_loss: 0.5314 - val_acc: 0.8303
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.015241579.
390/390 [==============================] - 53s 136ms/step - loss: 0.3169 - acc: 0.8874 - val_loss: 0.5214 - val_acc: 0.8363
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0148798452.
390/390 [==============================] - 53s 136ms/step - loss: 0.3075 - acc: 0.8916 - val_loss: 0.4523 - val_acc: 0.8517
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0145348837.
390/390 [==============================] - 53s 136ms/step - loss: 0.3049 - acc: 0.8919 - val_loss: 0.5511 - val_acc: 0.8228
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0142055544.
390/390 [==============================] - 53s 136ms/step - loss: 0.3040 - acc: 0.8922 - val_loss: 0.5650 - val_acc: 0.8236
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0138908182.
390/390 [==============================] - 53s 136ms/step - loss: 0.3034 - acc: 0.8939 - val_loss: 0.4585 - val_acc: 0.8478
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0135897262.
390/390 [==============================] - 53s 136ms/step - loss: 0.2950 - acc: 0.8967 - val_loss: 0.5053 - val_acc: 0.8377
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0133014099.
390/390 [==============================] - 53s 136ms/step - loss: 0.2907 - acc: 0.8975 - val_loss: 0.4550 - val_acc: 0.8511
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0130250733.
390/390 [==============================] - 53s 136ms/step - loss: 0.2891 - acc: 0.8979 - val_loss: 0.4337 - val_acc: 0.8580
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0127599847.
390/390 [==============================] - 53s 136ms/step - loss: 0.2861 - acc: 0.9000 - val_loss: 0.4373 - val_acc: 0.8536
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0125054711.
390/390 [==============================] - 53s 136ms/step - loss: 0.2824 - acc: 0.8997 - val_loss: 0.5045 - val_acc: 0.8413
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0122609122.
390/390 [==============================] - 53s 137ms/step - loss: 0.2791 - acc: 0.9019 - val_loss: 0.4263 - val_acc: 0.8606
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0120257351.
390/390 [==============================] - 53s 137ms/step - loss: 0.2751 - acc: 0.9032 - val_loss: 0.5123 - val_acc: 0.8406
Model took 2670.83 seconds to train


BEST Accuracy = 86.06%  in 49th Epoch
