import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.utils import to_categorical, get_custom_objects
from tensorflow.keras import backend as keras_bck
from tensorflow.keras import layers

########## Models ##########
def AV_Machine_Conv1D_LeNetLikeV1(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):
    """
    Total params: 133,954
    Trainable params: 133,910
    Non-trainable params: 44
    """
    
    if model_params_p is None:
        
        dropout_rate_l = 0.0
        dropout_rate_FC_l = 0.0

        reg_l2_l = 0.0
        reg_l2_FC_l = 0.0

    else:

        if "optuna_trial" in model_params_p:
            trial = model_params_p["optuna_trial"]

            dropout_rate_l = trial.suggest_float("dropout_rate_l", 0, 0.9)
            dropout_rate_FC_l = trial.suggest_float("dropout_rate_FC_l", 0, 0.9)
            reg_l2_l = trial.suggest_float("reg_l2_l", 0, 1.0)
            reg_l2_FC_l = trial.suggest_float("reg_l2_FC_l", 0, 1.0)

        if "mlflow" in model_params_p:
            mlflow = model_params_p["mlflow"]

            mlflow.log_param("dropout_rate_l", dropout_rate_l)
            mlflow.log_param("dropout_rate_FC_l", dropout_rate_FC_l)
            mlflow.log_param("reg_l2_l", reg_l2_l)
            mlflow.log_param("reg_l2_FC_l", reg_l2_FC_l)

    desc_l = "reg_l2_l =" + str(reg_l2_l) + ", reg_l2_FC_l =" + str(reg_l2_FC_l) + ", dropout_rate_l =" + str(dropout_rate_l) + ", dropout_rate_FC_l =" + str(dropout_rate_FC_l)

    model_l = AV_Model_Sequential()

    model_l.add(AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p)))

    AV_Layer_Conv1D_Blocks( model_p=model_l, 
                            N_Conv1D_p=1, 
                            filters_p=6,
                            kernel_size_p=5,
                            strides_p=1,
                            kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_l),
                            dropout_rate_p=dropout_rate_l)

    model_l.add(AV_Layer_AveragePooling1D(pool_size_p=2, strides_p=2))

    AV_Layer_Conv1D_Blocks( model_p=model_l, 
                            N_Conv1D_p=1, 
                            filters_p=16,
                            kernel_size_p=5,
                            strides_p=1,
                            kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_l),
                            dropout_rate_p=dropout_rate_l)

    model_l.add(AV_Layer_AveragePooling1D(pool_size_p=2, strides_p=2))

    model_l.add(AV_Layer_Flatten())           
    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_FC_l))               

    model_l.add(AV_Layer_Dense(units_p=120, kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_FC_l)))
    model_l.add(AV_Layer_Activation(activation_p="relu")) 
    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_FC_l))

    model_l.add(AV_Layer_Dense(units_p=84, kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_FC_l)))
    model_l.add(AV_Layer_Activation(activation_p="relu")) 
    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_FC_l))

    model_l.add(AV_Layer_Dense(units_p=nb_classes_p, kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_FC_l)))
    model_l.add(AV_Layer_Activation(activation_p="softmax"))

    return model_l, desc_l, "LeNetLikeV1"

def AV_Machine_Conv1D_Wang2016_ResNet(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):
    """
    Src. Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline by Wang et al. 2017
    """

    if model_params_p == "":
        reg_l2_l = 0.0
        dropout_rate_l = 0.0
    else:
        reg_l2_l = float(model_params_p.split("@")[0])
        dropout_rate_l = float(model_params_p.split("@")[1])
    
    desc_l = "reg_l2_l =" + str(reg_l2_l) + ", dropout_rate_l =" + str(dropout_rate_l)

    ## ResNet block ##
    x_l = AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p))

    y_l = AV_Layer_BatchNormalization(trainable_p=True)(x_l)

    n_feature_maps_l = 64
    y_l = AV_Layer_Conv1D_ResNet_Block( x_p=y_l, \
                                        filters_list_p=[n_feature_maps_l, n_feature_maps_l, n_feature_maps_l], \
                                        kernel_size_list_p=[8, 5, 3], \
                                        reg_l2_p=reg_l2_l, \
                                        name_p="block1")

    y_l = AV_Layer_Conv1D_ResNet_Block( x_p=y_l, \
                                        filters_list_p=[n_feature_maps_l*2, n_feature_maps_l*2, n_feature_maps_l*2], \
                                        kernel_size_list_p=[8, 5, 3], \
                                        reg_l2_p=reg_l2_l, \
                                        name_p="block2")

    y_l = AV_Layer_Conv1D_ResNet_Block( x_p=y_l, \
                                        filters_list_p=[n_feature_maps_l*2, n_feature_maps_l*2, n_feature_maps_l*2], \
                                        kernel_size_list_p=[8, 5, 3], \
                                        reg_l2_p=reg_l2_l, \
                                        name_p="block3")

    y_l = AV_Layer_GlobalAveragePooling1D()(y_l)
    y_l = AV_Layer_Dropout(rate_p=dropout_rate_l)(y_l)    
    y_l = AV_Layer_Dense(units_p=nb_classes_p)(y_l)
    y_l = AV_Layer_Activation(activation_p="softmax")(y_l)

    return keras.models.Model(inputs=x_l, outputs=y_l), desc_l, "Wang2016_ResNet"

def AV_Machine_Conv1D_vanPutten2018(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):
    """
    Trainable params: 1,281,670
    Yok:yes
    """
    
    dropout_rate_l = 0.0
    dropout_rate_FC_l = 0.0

    desc_l = "dropout_rate_l = " + str(dropout_rate_l) + ", dropout_rate_FC_l = " + str(dropout_rate_FC_l)

    model_l = AV_Model_Sequential()    

    model_l.add(AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p)))
           
    model_l.add(AV_Layer_Conv1D(filters_p=100,
                                kernel_size_p=3,
                                strides_p=1,
                                padding_p="same",
                                kernel_initializer_p="he_normal",
                                kernel_regularizer_p=None))
    model_l.add(AV_Layer_Activation(activation_p="relu"))
    model_l.add(AV_Layer_MaxPooling1D(pool_size_p=2))

    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_l))

    model_l.add(AV_Layer_Conv1D(filters_p=100,
                                kernel_size_p=3,
                                strides_p=1,
                                padding_p="same",
                                kernel_initializer_p="he_normal",
                                kernel_regularizer_p=None))
    model_l.add(AV_Layer_Activation(activation_p="relu"))
    model_l.add(AV_Layer_MaxPooling1D(pool_size_p=2))

    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_l))

    model_l.add(AV_Layer_Conv1D(filters_p=300,
                                kernel_size_p=3,
                                strides_p=1,
                                padding_p="same",
                                kernel_initializer_p="he_normal",
                                kernel_regularizer_p=None))
    model_l.add(AV_Layer_Activation(activation_p="relu"))
    model_l.add(AV_Layer_MaxPooling1D(pool_size_p=2))

    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_l))

    model_l.add(AV_Layer_Conv1D(filters_p=300,
                                kernel_size_p=7,
                                strides_p=1,
                                padding_p="same",
                                kernel_initializer_p="he_normal",
                                kernel_regularizer_p=None))
    model_l.add(AV_Layer_Activation(activation_p="relu"))
    model_l.add(AV_Layer_MaxPooling1D(pool_size_p=2))

    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_l))

    model_l.add(AV_Layer_Conv1D(filters_p=100,
                                kernel_size_p=3,
                                strides_p=1,
                                padding_p="same",
                                kernel_initializer_p="he_normal",
                                kernel_regularizer_p=None))
    model_l.add(AV_Layer_Activation(activation_p="relu"))

    model_l.add(AV_Layer_Conv1D(filters_p=100,
                                kernel_size_p=3,
                                strides_p=1,
                                padding_p="same",
                                kernel_initializer_p="he_normal",
                                kernel_regularizer_p=None))
    model_l.add(AV_Layer_Activation(activation_p="relu"))

    model_l.add(AV_Layer_Flatten())

    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_FC_l))

    model_l.add(AV_Layer_Dense(units_p=256))
    model_l.add(AV_Layer_Activation(activation_p="relu"))

    model_l.add(AV_Layer_Dropout(rate_p=dropout_rate_FC_l))

    model_l.add(AV_Layer_Dense(units_p=nb_classes_p))
    model_l.add(AV_Layer_Activation(activation_p="softmax"))

    return model_l, desc_l, "vanPutten2018"

def AV_Machine_Conv1D_Zagoruyko2016_WideResNet(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):
    """
    Trainable params: 67,906 for n_block_l = 2 and width_l = 1
    Trainable params: 198,722 for n_block_l = 6 and width_l = 1, Yok:yes
    Trainable params: 8,554,018 for n_block_l = 2 and width_l = 12
    Trainable params: 27,154,978 for n_block_l = 6 and width_l = 12, Yok:no
    """

    reg_l2_l = 0.0
    dropout_rate_l = 0.0
    n_block_l = 2 #+ 4 # the number of basic blocks in each group. Original paper investigates from 2 to 6.
    width_l = 1 + 5 + 6 # multiplier of the basic width. Orignal paper investigates from 1 to 12.

    desc_l = "reg_l2_l =" + str(reg_l2_l) + ", dropout_rate_l =" + str(dropout_rate_l) + ", n_block_l =" + str(n_block_l) + ", width_l =" + str(width_l)

    depth_l = 6*n_block_l + 4

    widths_l = [int(v * width_l) for v in (16, 32, 64)] # A list of no. of output's features w.r.t to each group.

    # assert (depth_l - 4) % 6 == 0 # depth should be 6n+4
    # n_block_l = (depth_l - 4) // 6 # the number of blocks in each group

    x_l = AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p))

    i_filters_l = 16
    y_l = AV_Layer_Conv1D(  filters_p=i_filters_l,
                            kernel_size_p=3,
                            padding_p="same",
                            use_bias_p=False, 
                            kernel_initializer_p="he_normal",
                            kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_l))(x_l) 

    y_l = AV_Layer_Conv1D_WideResNet_Group(y_l, n_block_p=n_block_l, o_filters_p=widths_l[0], kernel_size_p=3, stride_p=1, reg_l2_p=reg_l2_l, dropout_rate_p=dropout_rate_l)
    y_l = AV_Layer_Conv1D_WideResNet_Group(y_l, n_block_p=n_block_l, o_filters_p=widths_l[1], kernel_size_p=3, stride_p=2, reg_l2_p=reg_l2_l, dropout_rate_p=dropout_rate_l)
    y_l = AV_Layer_Conv1D_WideResNet_Group(y_l, n_block_p=n_block_l, o_filters_p=widths_l[2], kernel_size_p=3, stride_p=2, reg_l2_p=reg_l2_l, dropout_rate_p=dropout_rate_l)

    y_l = AV_Layer_BatchNormalization(trainable_p=True)(y_l)
    y_l = AV_Layer_Activation(activation_p="relu")(y_l)

    y_l = AV_Layer_AveragePooling1D(pool_size_p=8, strides_p=1)(y_l)

    y_l = AV_Layer_Flatten()(y_l)

    y_l = AV_Layer_Dense(units_p=nb_classes_p)(y_l)
    y_l = AV_Layer_Activation(activation_p="softmax")(y_l)

    return keras.models.Model(inputs=x_l, outputs=y_l), desc_l, "Zagoruyko2016_WideResNet"

def AV_Machine_Conv1D_Schirrmeister2017_shallowcovnet(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):  
    """
    Trainable params: 4,962
    """

    if model_params_p is None:

        reg_l2_l = 0.0
        dropout_rate_Conv_l = 0.5

    else:

        if "optuna_trial" in model_params_p:
            trial = model_params_p["optuna_trial"]

            reg_l2_l = trial.suggest_float("reg_l2_l", 0, 1.0)
            dropout_rate_Conv_l = trial.suggest_float("dropout_rate_Conv_l", 0, 0.9)

        if "mlflow" in model_params_p:
            mlflow = model_params_p["mlflow"]

            mlflow.log_param("reg_l2_l", reg_l2_l)
            mlflow.log_param("dropout_rate_Conv_l", dropout_rate_Conv_l)

    desc_l = "reg_l2_l =" + str(reg_l2_l) + ", dropout_rate_Conv_l =" + str(dropout_rate_Conv_l) 

    model_l = AV_Model_Sequential()    

    model_l.add(AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p)))

    model_l.add(AV_Layer_Conv1D(filters_p=40,
                                kernel_size_p=25,
                                strides_p=1,
                                padding_p='valid',
                                use_bias_p=False,
                                kernel_initializer_p='he_normal',
                                kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_l)))

    model_l.add(AV_Layer_BatchNormalization(trainable_p=True))

    model_l.add(AV_Layer_Activation(activation_p="square")) # original
    # model_l.add(AV_Layer_Activation(activation_p="relu"))     

    ## Pooling
    model_l.add(AV_Layer_MaxPooling1D(pool_size_p=75, strides_p=15))
    # model_l.add(AV_Layer_AveragePooling1D(pool_size_p=75, strides_p=15))
    
    model_l.add(AV_Layer_Activation(activation_p="safe_log")) # original

    ## Fully connected layer
    model_l.add(AV_Layer_Flatten())

    model_l.add(AV_Layer_Dropout(dropout_rate_Conv_l))

    model_l.add(AV_Layer_Dense(units_p=nb_classes_p))
    model_l.add(AV_Layer_Activation(activation_p="softmax"))

    return model_l, desc_l, "Schirrmeister2017_shallowcovnet"

def AV_Machine_Conv1D_Schirrmeister2017_deepcovnet(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):

    reg_l2_l = 0.0
    dropout_rate_Conv_l = 0.5

    desc_l = "reg_l2_l =" + str(reg_l2_l) + ", dropout_rate_Conv_l =" + str(dropout_rate_Conv_l) 

    model_l = AV_Model_Sequential()    

    model_l.add(AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p)))

    model_l.add(AV_Layer_Conv1D(filters_p=25,
                                kernel_size_p=10,
                                strides_p=1,
                                padding_p='valid',
                                use_bias_p=False,
                                kernel_initializer_p='he_normal',
                                kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_l)))

    model_l.add(AV_Layer_BatchNormalization(trainable_p=True))
    model_l.add(AV_Layer_Activation(activation_p="elu")) # original
    model_l.add(AV_Layer_MaxPooling1D(pool_size_p=3, strides_p=3))
    model_l.add(AV_Layer_Activation(activation_p="identity")) # original

    def add_conv_pool_block(model_p, filters_p, kernel_size_p, strides_p, reg_l2_p, dropout_rate_Conv_p):
        model_p.add(AV_Layer_Dropout(dropout_rate_Conv_p))        

        model_p.add(AV_Layer_Conv1D(filters_p=filters_p,
                                    kernel_size_p=kernel_size_p,
                                    strides_p=strides_p,
                                    padding_p='same',
                                    use_bias_p=False,
                                    kernel_initializer_p='he_normal',
                                    kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_p)))

        model_p.add(AV_Layer_BatchNormalization(trainable_p=True))

        model_p.add(AV_Layer_Activation(activation_p="elu")) # original

        model_p.add(AV_Layer_MaxPooling1D(pool_size_p=3, strides_p=3))

        model_p.add(AV_Layer_Activation(activation_p="identity")) # original

    add_conv_pool_block(model_l, 50, 10, 1, reg_l2_l, dropout_rate_Conv_l)
    add_conv_pool_block(model_l, 100, 10, 1, reg_l2_l, dropout_rate_Conv_l)
    add_conv_pool_block(model_l, 200, 10, 1, reg_l2_l, dropout_rate_Conv_l)

    ## Fully connected layers.
    model_l.add(AV_Layer_Flatten())

    model_l.add(AV_Layer_Dropout(dropout_rate_Conv_l))

    model_l.add(AV_Layer_Dense(units_p=nb_classes_p))
    model_l.add(AV_Layer_Activation(activation_p="softmax"))

    return model_l, desc_l, "Schirrmeister2017_deepcovnet"

def AV_Machine_Conv1D_MobileNet(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):
    """
    Trainable params: 3,178,786
    Yok:no
    """

    alpha_l = 1.0
    depth_multiplier_l = 1
    dropout_l = 0.001

    desc_l = ""

    def MobileNet(alpha=1.0,
                  depth_multiplier=1,
                  dropout=1e-3,
                  input_tensor=None,
                  pooling=None,
                  **kwargs):
        """Instantiates the MobileNet architecture.

        # Arguments
            alpha: controls the width of the network. This is known as the
                width multiplier in the MobileNet paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            depth_multiplier: depth multiplier for depthwise convolution. This
                is called the resolution multiplier in the MobileNet paper.
            dropout: dropout rate
            input_tensor: optional Keras tensor (i.e. output of
                `layers.Input()`)
                to use as image input for the model.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model
                    will be the 4D tensor output of the
                    last convolutional block.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a
                    2D tensor.
                - `max` means that global max pooling will
                    be applied.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
        """

        x = _conv_block(input_tensor, 32, alpha, strides=2)
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=2, block_id=2)
        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=2, block_id=4)
        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=2, block_id=6)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=2, block_id=12)
        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

        if pooling == 'avg':
            x = layers.GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling1D()(x)

        # Create model.
        model = keras.models.Model(input_tensor, x, name='mobilenet')

        return model


    def _conv_block(inputs, filters, alpha, kernel=3, strides=1):
        """Adds an initial convolution layer (with batch normalization and relu6).

        # Arguments
            inputs: Input tensor of shape `(rows, cols, 3)`
                (with `channels_last` data format) or
                (3, rows, cols) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(224, 224, 3)` would be one valid value.
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.

        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(samples, filters, new_rows, new_cols)`
            if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, new_rows, new_cols, filters)`
            if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = -1
        filters = int(filters * alpha)
        x = layers.ZeroPadding1D(padding=((0, 1)), name='conv1_pad')(inputs)
        x = layers.Conv1D(filters, kernel,
                          padding='valid',
                          use_bias=False,
                          strides=strides,
                          name='conv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
        return layers.ReLU(6., name='conv1_relu')(x)


    def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=1, block_id=1):
        """Adds a depthwise convolution block.

        A depthwise convolution block consists of a depthwise conv,
        batch normalization, relu6, pointwise convolution,
        batch normalization and relu6 activation.

        # Arguments
            inputs: Input tensor of shape `(rows, cols, channels)`
                (with `channels_last` data format) or
                (channels, rows, cols) (with `channels_first` data format).
            pointwise_conv_filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the pointwise convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            depth_multiplier: The number of depthwise convolution output channels
                for each input channel.
                The total number of depthwise convolution output
                channels will be equal to `filters_in * depth_multiplier`.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            block_id: Integer, a unique identification designating
                the block number.

        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)`
            if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)`
            if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        if strides == 1:
            x = inputs
        else:
            x = layers.ZeroPadding1D( ((0, 1)) , name='conv_pad_%d' % block_id)(inputs)

        x = AV_keras_DepthwiseConv1D(   x_p=x,
                                        kernel_size_p=3,
                                        strides_p=strides,
                                        padding_p='same' if strides == 1 else 'valid',
                                        depth_multiplier_p=depth_multiplier,
                                        use_bias_p=False,
                                        name='conv_dw_%d' % block_id)


        # x = layers.DepthwiseConv2D((3, 3),
        #                            padding='same' if strides == (1, 1) else 'valid',
        #                            depth_multiplier=depth_multiplier,
        #                            strides=strides,
        #                            use_bias=False,
        #                            name='conv_dw_%d' % block_id)(x)

        x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
        x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

        x = layers.Conv1D(pointwise_conv_filters, 1,
                          padding='same',
                          use_bias=False,
                          strides=1,
                          name='conv_pw_%d' % block_id)(x)
        x = layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
        return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    ## Core
    x_l = AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p))

    y_l = MobileNet(alpha=alpha_l, depth_multiplier=depth_multiplier_l, dropout=1e-3, input_tensor=x_l, pooling="avg").output

    y_l = AV_Layer_Dropout(rate_p=dropout_l)(y_l)
    y_l = AV_Layer_Dense(units_p=nb_classes_p)(y_l)    
    y_l = AV_Layer_Activation(activation_p="softmax")(y_l)

    return keras.models.Model(inputs=x_l, outputs=y_l), desc_l, "MobileNet" 

def AV_Machine_Conv1D_Xception(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=None):
    """
    Trainable params: 20,659,994
    Pre: no, Nam: yes
    """

    desc_l = ""

    def Xception(input_tensor=None, pooling=None):
        """Instantiates the Xception architecture.

        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in your Keras config at `~/.keras/keras.json`.

        Note that the default input image size for this model is 299x299.

        # Arguments
            input_tensor: optional Keras tensor
                (i.e. output of `layers.Input()`)
                to use as image input for the model.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
        """
        
        channel_axis = -1

        x = layers.Conv1D(32, 3, strides=2, use_bias=False, name='block1_conv1')(input_tensor)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = layers.Activation('relu', name='block1_conv1_act')(x)
        x = layers.Conv1D(64, 3, use_bias=False, name='block1_conv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = layers.Activation('relu', name='block1_conv2_act')(x)

        residual = layers.Conv1D(128, 1, strides=2, padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.SeparableConv1D(128, 3, padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block2_sepconv2_act')(x)
        x = layers.SeparableConv1D(128, 3, padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same', name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv1D(256, 1, strides=2, padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = layers.SeparableConv1D(256, 3, padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block3_sepconv2_act')(x)
        x = layers.SeparableConv1D(256, 3, padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same', name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv1D(728, 1, strides=2, padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block4_sepconv2_act')(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same', name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

            x = layers.add([x, residual])

        residual = layers.Conv1D(1024, 1, strides=2, padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block13_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = layers.SeparableConv1D(1024, 3, padding='same', use_bias=False, name='block13_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same', name='block13_pool')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv1D(1536, 3, padding='same', use_bias=False, name='block14_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv1D(2048, 3, padding='same', use_bias=False, name='block14_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv2_act')(x)

        if pooling == 'avg':
            x = layers.GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling1D()(x)

        # Create model.
        model = keras.models.Model(input_tensor, x, name='xception')

        return model      

    x_l = AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p))

    y_l = Xception(input_tensor=x_l, pooling="avg").output

    y_l = AV_Layer_Dense(units_p=nb_classes_p)(y_l)    
    y_l = AV_Layer_Activation(activation_p="softmax")(y_l)

    return keras.models.Model(inputs=x_l, outputs=y_l), desc_l, "Xception"

########## Trainning functions ##########
def AV_Model_compile(   model_p, 
                        optimizer_p,                # String (name of optimizer) or optimizer instance. See optimizers. 
                        loss_p=None,                # String (name of objective function) or objective function or Loss instance. See losses. 
                                                    # If the model has multiple outputs, you can use a different loss on each output by passing 
                                                    # a dictionary or a list of losses. The loss value that will be minimized by the model will then 
                                                    # be the sum of all individual losses.
                        metrics_p=None,             # List of metrics to be evaluated by the model during training and testing. Typically you will 
                                                    # use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output 
                                                    # model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}. 
                                                    # You can also pass a list (len = len(outputs)) of lists of metrics such as metrics=[['accuracy'], 
                                                    # ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']].
                        loss_weights_p=None,        # Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss 
                                                    # contributions of different model outputs. The loss value that will be minimized by the model will 
                                                    # then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. 
                                                    # If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is expected 
                                                    # to map output names (strings) to scalar coefficients. 
                        sample_weight_mode_p=None,  # If you need to do timestep-wise sample weighting (2D weights), set this to "temporal". 
                                                    # None defaults to sample-wise weights (1D). If the model has multiple outputs, you can use 
                                                    # a different sample_weight_mode on each output by passing a dictionary or a list of modes.
                        weighted_metrics_p=None,    # List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                        target_tensors_p=None       # By default, Keras will create placeholders for the model's target, which will be fed with the 
                                                    # target data during training. If instead you would like to use your own target tensors (in turn, 
                                                    # Keras will not expect external Numpy data for these targets at training time), you can specify 
                                                    # them via the target_tensors argument. It can be a single tensor (for a single-output model), 
                                                    # a list of tensors, or a dict mapping output names to target tensors.
                        ):
    """
    Configures the model for training.
    """

    model_p.compile(optimizer=optimizer_p, 
                    loss=loss_p, 
                    metrics=metrics_p, 
                    loss_weights=loss_weights_p, 
                    sample_weight_mode=sample_weight_mode_p, 
                    weighted_metrics=weighted_metrics_p, 
                    target_tensors=target_tensors_p)


def AV_Model_train_allbatches(  model_p,
                                x_p=None,                       # Input data. It could be:
                                                                #   A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                #   A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                #   A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                #   None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                y_p=None,                       # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                batch_size_p=None,              # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                epochs_p=1,                     # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                callbacks_p=None,               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                validation_data_p=None,         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                ):                                
    """
    This method is different from AV_Model_train_batchBybatch() in such a way that if we can load all trainning and validation data inside the memory all at once.
    Another difference is that there is no data augmentation in this function, i.e. we train on the raw data.
    """

    history_l = model_p.fit(x=x_p, 
                            y=y_p, 
                            batch_size=batch_size_p, 
                            epochs=epochs_p, 
                            verbose=verbose_p, 
                            callbacks=callbacks_p, 
                            validation_split=validation_split_p, 
                            validation_data=validation_data_p, 
                            shuffle=shuffle_p, 
                            class_weight=class_weight_p, 
                            sample_weight=sample_weight_p, 
                            initial_epoch=initial_epoch_p, 
                            steps_per_epoch=steps_per_epoch_p, 
                            validation_steps=validation_steps_p, 
                            validation_freq=validation_freq_p, 
                            max_queue_size=max_queue_size_p, 
                            workers=workers_p, 
                            use_multiprocessing=use_multiprocessing_p)

    return history_l

########## ML legos ##########
def identity(x):
    return x

def square(x):
    return keras_bck.square(x)

def safe_log(x):
    return keras_bck.log(keras_bck.clip(x, min_value=np.finfo(np.float32).eps, max_value=None)) # Prevent log(0)

def swish(x, beta_p = 1):
    return (x * sigmoid(beta_p * x))    

get_custom_objects().update({"identity": identity})
get_custom_objects().update({"square": square})
get_custom_objects().update({"safe_log": safe_log})
get_custom_objects().update({"swish": swish})

def AV_Model_Sequential():
    return keras.models.Sequential()

def AV_Layer_Input( input_shape_p=None,     # Shape tuple (not including the batch axis), or TensorShape instance (not including the batch axis).
                    batch_size_p=None,      # Optional input batch size (integer or None).
                    dtype_p=None,           # Datatype of the input.
                    input_tensor_p=None,    # Optional tensor to use as layer input instead of creating a placeholder.
                    sparse_p=False,         # Boolean, whether the placeholder created is meant to be sparse.
                    name_p=None,            # Name of the layer (string).
                    ragged_p=False):         # Boolean, whether the placeholder created is meant to be ragged. In this case, values of 'None' in the 'shape' argument 
                                            # represent ragged dimensions. For more information about RaggedTensors, see https://www.tensorflow.org/guide/ragged_tensors.

    return keras.Input( shape=input_shape_p,
                        batch_size=batch_size_p,
                        name=name_p,
                        dtype=dtype_p,
                        sparse=sparse_p,
                        tensor=None,
                        ragged=ragged_p)


def AV_Layer_Conv1D_Blocks( model_p, 
                            N_Conv1D_p, 
                            filters_p,
                            kernel_size_p=3,
                            strides_p=1,
                            kernel_regularizer_p=None,
                            activation_p="relu",
                            dropout_rate_p=0.0):

    for i_l in range(N_Conv1D_p):
        model_p.add(AV_Layer_Conv1D(filters_p=filters_p,
                                    kernel_size_p=kernel_size_p,
                                    strides_p=strides_p,
                                    padding_p="same",
                                    kernel_initializer_p="he_normal",
                                    kernel_regularizer_p=kernel_regularizer_p))

        model_p.add(AV_Layer_BatchNormalization(trainable_p=True))   

        model_p.add(AV_Layer_Activation(activation_p=activation_p)) 

        model_p.add(AV_Layer_Dropout(rate_p=dropout_rate_p))    


def AV_Layer_Conv1D(filters_p,                              # Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
                    kernel_size_p,                          # An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
                    strides_p=1,                            # An integer or tuple/list of a single integer, specifying the stride length of the convolution. 
                                                            # Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
                    padding_p='valid',                      # One of "valid", "causal" or "same" (case-insensitive). "causal" results in causal (dilated) convolutions, 
                                                            # e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order. 
                                                            # See WaveNet: A Generative Model for Raw Audio, section 2.1.
                    data_format_p='channels_last',          # A string, one of channels_last (default) or channels_first.
                    dilation_rate_p=1,                      # An integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. 
                                                            # Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
                    activation_p=None,                      # Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
                    use_bias_p=True,                        # Boolean, whether the layer uses a bias vector.
                    kernel_initializer_p='glorot_uniform',  # Initializer for the kernel weights matrix.
                    bias_initializer_p='zeros',             # Initializer for the bias vector.
                    kernel_regularizer_p=None,              # Regularizer function applied to the kernel weights matrix.
                    bias_regularizer_p=None,                # Regularizer function applied to the bias vector.
                    activity_regularizer_p=None,            # Regularizer function applied to the output of the layer (its "activation").
                    kernel_constraint_p=None,               # Constraint function applied to the kernel matrix.
                    bias_constraint_p=None,                 # Constraint function applied to the bias vector.
                    name_p=None):

    return keras.layers.Conv1D( filters=filters_p, 
                                kernel_size=kernel_size_p, 
                                strides=strides_p, 
                                padding=padding_p,
                                data_format=data_format_p,
                                dilation_rate=dilation_rate_p, 
                                activation=activation_p, 
                                use_bias=use_bias_p,
                                kernel_initializer=kernel_initializer_p, 
                                bias_initializer=bias_initializer_p,
                                kernel_regularizer=kernel_regularizer_p, 
                                bias_regularizer=bias_regularizer_p, 
                                activity_regularizer=activity_regularizer_p,
                                kernel_constraint=kernel_constraint_p, 
                                bias_constraint=bias_constraint_p,
                                name=name_p)

def AV_Layer_BatchNormalization(    axis_p=-1,                              # Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization.
                                    momentum_p=0.99,                        # Momentum for the moving average.
                                    epsilon_p=0.001,                        # Small float added to variance to avoid dividing by zero.
                                    center_p=True,                          # If True, add offset of beta to normalized tensor. If False, beta is ignored.
                                    scale_p=True,                           # If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer.
                                    beta_initializer_p='zeros',             # Initializer for the beta weight.
                                    gamma_initializer_p='ones',             # Initializer for the gamma weight.
                                    moving_mean_initializer_p='zeros',      # Initializer for the moving mean.
                                    moving_variance_initializer_p='ones',   # Initializer for the moving variance.
                                    beta_regularizer_p=None,                # Optional regularizer for the beta weight.
                                    gamma_regularizer_p=None,               # Optional regularizer for the gamma weight.
                                    beta_constraint_p=None,                 # Optional constraint for the beta weight.
                                    gamma_constraint_p=None,                # Optional constraint for the gamma weight.
                                    renorm_p=False,                         # Whether to use Batch Renormalization. This adds extra variables during training. The inference is the same for either value of this parameter.
                                    renorm_clipping_p=None,                 # A dictionary that may map keys 'rmax', 'rmin', 'dmax' to scalar Tensors used to clip the renorm correction. The correction (r, d) is used as corrected_value = normalized_value * r + d, with r clipped to [rmin, rmax], and d to [-dmax, dmax]. Missing rmax, rmin, dmax are set to inf, 0, inf, respectively.
                                    renorm_momentum_p=0.99,                 # Momentum used to update the moving means and standard deviations with renorm. Unlike momentum, this affects training and should be neither too small (which would add noise) nor too large (which would give stale estimates). Note that momentum is still applied to get the means and variances for inference.
                                    fused_p=None,                           # if True, use a faster, fused implementation, or raise a ValueError if the fused implementation cannot be used. If None, use the faster implementation if possible. If False, do not used the fused implementation.
                                    trainable_p=True,                       # Boolean, if True the variables will be marked as trainable.
                                    virtual_batch_size_p=None,              # An int. By default, virtual_batch_size is None, which means batch normalization is performed across the whole batch. When virtual_batch_size is not None, instead perform "Ghost Batch Normalization", which creates virtual sub-batches which are each normalized separately (with shared gamma, beta, and moving statistics). Must divide the actual batch size during execution.
                                    adjustment_p=None,                      # A function taking the Tensor containing the (dynamic) shape of the input tensor and returning a pair (scale, bias) to apply to the normalized values (before gamma and beta), only during training. For example, if axis==-1, adjustment = lambda shape: ( tf.random.uniform(shape[-1:], 0.93, 1.07), tf.random.uniform(shape[-1:], -0.1, 0.1)) will scale the normalized value by up to 7% up or down, then shift the result by up to 0.1 (with independent scaling and bias for each feature but shared across all examples), and finally apply gamma and/or beta. If None, no adjustment is applied. Cannot be specified if virtual_batch_size is specified.
                                    name_p=None
                                    ):
    """
    Src.: https://stackoverflow.com/questions/57692323/non-trainable-parameters-params-in-keras-model-is-calculated

    Assume the layer before the batchnormalization outputs N_feature_maps_l. BatchNormalization generates 4*N_feature_maps_l parameters, of which 
    2*N_feature_maps_l non-trainnable parameters and 2*N_feature_maps_l trainnable parameters.
    """    

    return keras.layers.BatchNormalization( axis=axis_p, 
                                            momentum=momentum_p, 
                                            epsilon=epsilon_p, 
                                            center=center_p, 
                                            scale=scale_p,
                                            beta_initializer=beta_initializer_p, 
                                            gamma_initializer=gamma_initializer_p,
                                            moving_mean_initializer=moving_mean_initializer_p, 
                                            moving_variance_initializer=moving_variance_initializer_p,
                                            beta_regularizer=beta_regularizer_p, 
                                            gamma_regularizer=gamma_regularizer_p, 
                                            beta_constraint=beta_constraint_p,
                                            gamma_constraint=gamma_constraint_p, 
                                            renorm=renorm_p, 
                                            renorm_clipping=renorm_clipping_p, 
                                            renorm_momentum=renorm_momentum_p,
                                            fused=fused_p, 
                                            trainable=trainable_p, 
                                            virtual_batch_size=virtual_batch_size_p, 
                                            adjustment=adjustment_p, 
                                            name=name_p)

def AV_Layer_Activation(activation_p="relu", name_p=None, ret_kernel_initializer=False):
    """
    See https://mlfromscratch.com/activation-functions-explained/#/
    """

    if not ret_kernel_initializer:
        ## ReLU solves vanishing gradient by defining two values of gradients either 0 or 1 (nothing close to 0!!!). But, it can be trapped in the dead state (0 gradient).
        ## The way out of the dead state is leaky relu. 
        ## If we want to use Dropout for SELU, we shall use AlphaDropout (see https://mlfromscratch.com/activation-functions-explained/#/)  

        if activation_p == "square":
            return keras.layers.Activation(square, name=name_p)
        elif activation_p == "safe_log":
            return keras.layers.Activation(safe_log, name=name_p)
        elif activation_p == "swish":
            return keras.layers.Activation(swish, name=name_p)                        
        elif activation_p == "leaky_relu":        
            return keras.layers.LeakyReLU(alpha=0.3)
        else:
            return keras.layers.Activation(activation=activation_p, name=name_p)
    else:
        ## See https://towardsdatascience.com/how-to-initialize-a-neural-network-27564cfb5ffc
        ## Assume the output of the previous layer is i1, ..., iN. The outputs are given to the next layer and yield o1, ..., oM (after activation). We want o1, ..., oM all together forms the normal distribution (zero mean, unit std).
        ## If o1, ..., oM follows the normal distribution, the output of the forward layers will not blow up and vanish.
        ## It also means that the input of the deep network should be normalized before.

        if activation_p == "selu": 
            return "lecun_normal"
        elif (activation_p == "relu"):
            ## See https://towardsdatascience.com/how-to-initialize-a-neural-network-27564cfb5ffc
            ## Problem of using he_normal
            ## 1. After relu, the mean of the output is not 0 because relu removes all negative inputs.
            ## 2. It works only relu.
            ## 3. STD of the outputs is close to 1 (not exactly 1).
            return "he_normal" 
        elif (activation_p == "swish") or (activation_p == "elu") or (activation_p == "leaky_relu"):
            return "he_normal"
        else:
            return "glorot_uniform"   


def AV_Layer_AveragePooling1D(  pool_size_p=2,      # Integer, size of the average pooling windows
                                strides_p=None,     # Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
                                padding_p="valid"   # One of "valid" or "same" (case-insensitive).
                                ):

    return keras.layers.AveragePooling1D(pool_size=pool_size_p, strides=strides_p, padding=padding_p, data_format="channels_last")


def AV_Layer_Flatten(   data_format_p=None  # A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
                                            # The purpose of this argument is to preserve weight ordering when switching a model from one data format 
                                            # to another. channels_last corresponds to inputs with shape (batch, ..., channels) while channels_first 
                                            # corresponds to inputs with shape (batch, channels, ...). It defaults to the image_data_format value found 
                                            # in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
                        ):

    return keras.layers.Flatten(data_format=data_format_p)


def AV_Layer_Dropout(   rate_p,                 # Float, between 0 and 1. Fraction of the input units to drop.
                        noise_shape_p=None,     # 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. 
                                                # For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be 
                                                # the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
                        seed_p=None             # A Python integer to use as random seed.
                        ):
    """
    This dropout layer will not retain in the prediction (model.predict) or test phase.
    To test the dropout,
        r = 0.4
        print(1/(1-r))
        x = tf.keras.Input(shape=(32,))
        y = tf.keras.layers.Dropout(rate=r)(x)

        iterate_l = keras_bck.function(inputs=[x, keras_bck.learning_phase()], outputs=[y])

        input_l = AV_Ones((32,))
        print(iterate_l([input_l, 0]))
    """
    
    return keras.layers.Dropout(rate=rate_p, noise_shape=noise_shape_p, seed=seed_p)

def AV_Layer_Dense( units_p,                                # Positive integer, dimensionality of the output space.
                    activation_p=None,                      # Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
                    use_bias_p=True,                        # Boolean, whether the layer uses a bias vector.
                    kernel_initializer_p='glorot_uniform',  # Initializer for the kernel weights matrix (see initializers).
                    bias_initializer_p='zeros',             # Initializer for the bias vector (see initializers).
                    kernel_regularizer_p=None,              # Regularizer function applied to the kernel weights matrix (see regularizer).
                    bias_regularizer_p=None,                # Regularizer function applied to the bias vector (see regularizer).
                    activity_regularizer_p=None,            # Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
                    kernel_constraint_p=None,               # Constraint function applied to the kernel weights matrix (see constraints).
                    bias_constraint_p=None                  # Constraint function applied to the bias vector (see constraints).
                    ):

    return keras.layers.Dense(  units=units_p, 
                                activation=activation_p, 
                                use_bias=use_bias_p, 
                                kernel_initializer=kernel_initializer_p, 
                                bias_initializer=bias_initializer_p, 
                                kernel_regularizer=kernel_regularizer_p, 
                                bias_regularizer=bias_regularizer_p, 
                                activity_regularizer=activity_regularizer_p, 
                                kernel_constraint=kernel_constraint_p, 
                                bias_constraint=bias_constraint_p)

def AV_Layer_Conv1D_ResNet_Block(x_p, filters_list_p, kernel_size_list_p, reg_l2_p, name_p, activation_p="relu"):

    assert len(filters_list_p) == len(kernel_size_list_p)

    kernel_initializer_l = AV_Layer_Activation(activation_p=activation_p, ret_kernel_initializer=True)

    N_Conv1D_layers_l = len(filters_list_p)

    ## Direct path of x_p
    conv_x_l = x_p
    for i_l in range(N_Conv1D_layers_l - 1):
        conv_x_l = AV_Layer_Conv1D( filters_p=filters_list_p[i_l],
                                    kernel_size_p=kernel_size_list_p[i_l],
                                    strides_p=1,
                                    padding_p="same",
                                    use_bias_p=False,
                                    kernel_initializer_p=kernel_initializer_l,
                                    kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_p),
                                    name_p=name_p + "_direct_sub_conv1D_" + str(i_l))(conv_x_l)
        conv_x_l = AV_Layer_BatchNormalization(trainable_p=True, name_p=name_p + "_direct_sub_BN_" + str(i_l))(conv_x_l)
        conv_x_l = AV_Layer_Activation(activation_p=activation_p, name_p=name_p + "_direct_sub_Act_" + str(i_l))(conv_x_l)


    conv_x_l = AV_Layer_Conv1D( filters_p=filters_list_p[N_Conv1D_layers_l - 1],
                                kernel_size_p=kernel_size_list_p[N_Conv1D_layers_l - 1],
                                strides_p=1,
                                padding_p="same",
                                use_bias_p=False,
                                kernel_initializer_p=kernel_initializer_l,
                                kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_p),
                                name_p=name_p + "_direct_main_conv1D")(conv_x_l)
    conv_x_l = AV_Layer_BatchNormalization(trainable_p=True, name_p=name_p + "_direct_main_BN")(conv_x_l)

    ## Shortcut of x_p
    is_expand_channels_l = not (x_p.shape[-1] == filters_list_p[N_Conv1D_layers_l - 1])
    if is_expand_channels_l:
        shortcut_x_p = AV_Layer_Conv1D( filters_p=filters_list_p[N_Conv1D_layers_l - 1],
                                        kernel_size_p=1,
                                        strides_p=1,
                                        padding_p="same",
                                        use_bias_p=False,
                                        kernel_initializer_p=kernel_initializer_l,
                                        kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_p),
                                        name_p=name_p + "_shortcut_conv1D")(x_p)

        shortcut_x_p = AV_Layer_BatchNormalization(trainable_p=True, name_p=name_p + "_shortcut_BN")(shortcut_x_p)
    else:
        shortcut_x_p = AV_Layer_BatchNormalization(trainable_p=True, name_p=name_p + "_shortcut_BN")(x_p)
    
    ## Merging direct path and shortcut
    y_l = AV_Layer_Add([shortcut_x_p, conv_x_l])
    y_l = AV_Layer_Activation(activation_p=activation_p, name_p=name_p + "_Act")(y_l)

    return y_l

def AV_Layer_Add(layers_list_p):

    return keras.layers.Add()(layers_list_p)

def AV_Layer_GlobalAveragePooling1D():
    """
        For each feature dimension, it takes average among all time steps.
    """

    return keras.layers.GlobalAveragePooling1D(data_format="channels_last")

def AV_Layer_MaxPooling1D(  pool_size_p=2,      # Integer, size of the max pooling windows.
                            strides_p=None,     # Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
                            padding_p="valid"   # One of "valid" or "same" (case-insensitive)
                            ):

    return keras.layers.MaxPooling1D(pool_size=pool_size_p, strides=strides_p, padding=padding_p, data_format="channels_last")

def AV_Layer_Conv1D_WideResNet_Group(x_p, n_block_p, o_filters_p, kernel_size_p, stride_p, reg_l2_p, dropout_rate_p):

    for i_l in range(n_block_p):
        x_p = AV_Layer_Conv1D_WideResNet_Block(x_p, o_filters_p, kernel_size_p, stride_p if i_l == 0 else 1, reg_l2_p, dropout_rate_p)

    return x_p

def AV_Layer_Conv1D_WideResNet_Block(x_p, o_filters_p, kernel_size_p, stride_p, reg_l2_p, dropout_rate_p):

    i_filters_l = x_p.shape[-1]


    BNActConv_blocks_l = AV_Layer_Conv1D_BNActConv_Blocks(x_p, filters_p=o_filters_p, kernel_size_p=[kernel_size_p, kernel_size_p], stride_p=[stride_p, 1], reg_l2_p=reg_l2_p, dropout_rate_p=dropout_rate_p)


    ## Shortcut of x_p
    is_expand_channels_l = not (i_filters_l == o_filters_p)
    if is_expand_channels_l:

        shortcut_x_p = AV_Layer_Conv1D( filters_p=o_filters_p,
                                        kernel_size_p=1,
                                        strides_p=stride_p,
                                        padding_p="same",
                                        kernel_initializer_p="he_normal",
                                        kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_p))(x_p)

        y_l = AV_Layer_Add([BNActConv_blocks_l, shortcut_x_p])

    else:

        y_l = AV_Layer_Add([BNActConv_blocks_l, x_p])

    return y_l

def AV_Layer_Conv1D_BNActConv_Blocks(x_p, filters_p, kernel_size_p, stride_p, reg_l2_p, activation_p="relu", dropout_rate_p=0.0):
    """
    Params:
        kernel_size_p   : a list of integers of the kernel's size in each block. Assume square kernel size.
        stride_p        : a list of integers of the stride in each block. Assume square stride.
    """

    for i_l, _ in enumerate(kernel_size_p):

        x_p = AV_Layer_BatchNormalization(trainable_p=True)(x_p)

        x_p = AV_Layer_Activation(activation_p=activation_p)(x_p)

        if i_l > 0:
            x_p = AV_Layer_Dropout(rate_p=dropout_rate_p)(x_p)

        x_p = AV_Layer_Conv1D(  filters_p=filters_p,
                                kernel_size_p=kernel_size_p[i_l],
                                strides_p=stride_p[i_l],
                                padding_p="same",
                                use_bias_p=False, # we set it to False because BN takes care of the bias.
                                kernel_initializer_p="he_normal",
                                kernel_regularizer_p=keras.regularizers.l2(l=reg_l2_p))(x_p)

    return x_p


def AV_keras_DepthwiseConv1D(   x_p,
                                kernel_size_p,
                                strides_p=1,
                                padding_p='valid',
                                depth_multiplier_p=1,
                                data_format_p=None,
                                activation_p=None,
                                use_bias_p=True,
                                depthwise_initializer_p='glorot_uniform',
                                bias_initializer_p='zeros',
                                depthwise_regularizer_p=None,
                                bias_regularizer_p=None,
                                activity_regularizer_p=None,
                                depthwise_constraint_p=None,
                                bias_constraint_p=None,
                                **kwargs_p):
    """
    Params:
        x_p : A tensor has the following structure (BATCH, LENGTH, RGB). If we have only one time serie, BATCH equals 1.
    """

    ## Expand x_p to have (BATCH, 1, LENGTH, RGB)    
    x_l = keras_bck.expand_dims(x_p, axis=1)

    ## Do DepthwiseConv2D
    x_l = layers.DepthwiseConv2D(   kernel_size=(1, kernel_size_p),
                                    strides=(1, strides_p),
                                    padding=padding_p,
                                    depth_multiplier=depth_multiplier_p,
                                    data_format=data_format_p,
                                    activation=activation_p,
                                    use_bias=use_bias_p,
                                    depthwise_initializer=depthwise_initializer_p,
                                    bias_initializer=bias_initializer_p,
                                    depthwise_regularizer=depthwise_regularizer_p,
                                    bias_regularizer=bias_regularizer_p,
                                    activity_regularizer=activity_regularizer_p,
                                    depthwise_constraint=depthwise_constraint_p,
                                    bias_constraint=bias_constraint_p,
                                    **kwargs_p)(x_l)    

    ## Reduce to (BATCH, LENGTH, RGB)
    x_l = keras_bck.squeeze(x_l, axis=1)

    return x_l