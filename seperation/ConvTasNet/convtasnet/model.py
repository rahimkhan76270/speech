import tensorflow as tf  
from tensorflow.keras import Model,Sequential # type:ignore
from tensorflow.keras.layers import Conv1D,PReLU,BatchNormalization,Conv1DTranspose,LayerNormalization,ReLU # type:ignore


class ConvResBlock(Model):
    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(ConvResBlock,self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.causal=causal
        self.in_channels=in_channels
        self.conv1x1=Conv1D(
            filters=self.out_channels,
            kernel_size=1,
            data_format='channels_first',
            use_bias=False
        )
        self.PReLU_1=PReLU(shared_axes=[2])
        self.norm1=BatchNormalization(axis=1)
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        self.dwconv=Conv1D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            groups=self.out_channels,
            padding='same',
            dilation_rate=dilation,
            data_format='channels_first',
            use_bias=False
        )
        self.PReLU_2=PReLU(shared_axes=[2])
        self.norm2=BatchNormalization(axis=1)
        self.sc_conv=Conv1D(
            filters=self.in_channels,
            kernel_size=1,
            data_format='channels_first',
            use_bias=True
        )
    def call(self,x):
        c=self.conv1x1(x)
        c=self.PReLU_1(c)
        c=self.norm1(c)
        c=self.dwconv(c)
        if self.causal:
            c=c[:,:,:-self.pad]
        c=self.sc_conv(c)
        return c+x
    def build(self,input_shape):
        super(ConvResBlock,self).build(input_shape)

class Encoder(Model):
    def __init__(self,out_channels=512,
                 kernel_size=16):
        super(Encoder,self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.conv1=Conv1D(filters=self.out_channels,
                        kernel_size=self.kernel_size,
                        strides=self.kernel_size//2,
                        padding='valid',
                        data_format='channels_first')

    def call(self,x):
        y=self.conv1(x)
        return y

    def build(self,input_shape):
        super(Encoder,self).build(input_shape)

class Decoder(Model):
    def __init__(self,
                 out_channels=1,
                 kernel_size=16,
                 strides=8):
        super(Decoder,self).__init__()
        self.conv_trans1d=Conv1DTranspose(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            data_format='channels_first'
        )

    def call(self,x):
        x=self.conv_trans1d(x)
        return x

    def build(self,input_shape):
        super(Decoder,self).build(input_shape)

class SequentialBlock(Model):
    def __init__(self,num_blocks,**block_args):
        super(SequentialBlock,self).__init__()
        self.block_list=[ConvResBlock(**block_args,dilation=2**i) for i in range(num_blocks)]
        self.seq_block=Sequential(self.block_list)

    def call(self,x):
        x=self.seq_block(x)
        return x

    def build(self,input_shape):
        super(SequentialBlock,self).build(input_shape)

class SeparationBlock(Model):
    def __init__(self,num_repeat,**block_args):
        super(SeparationBlock,self).__init__()
        self.seq_repeat=Sequential([SequentialBlock(**block_args) for _ in range(num_repeat)])

    def call(self,x):
        x=self.seq_repeat(x)
        return x

    def build(self,input_shape):
        super(SeparationBlock,self).build(input_shape)

class ConvTasNet(Model):
    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 num_spks=2,
                 causal=False):
        super(ConvTasNet,self).__init__()
        self.encoder=Encoder(
            out_channels=N,
            kernel_size=L
        )
        self.layer_norm=LayerNormalization(axis=1)
        self.bottle_neck=Conv1D(
            filters=B,
            kernel_size=1,
            data_format='channels_first'
        )
        self.separation=SeparationBlock(
            num_repeat=R,
            num_blocks=X,
            in_channels=B,
            out_channels=H,
            kernel_size=P,
            causal=causal
        )
        self.gen_masks=Conv1D(
            filters=N*num_spks,
            kernel_size=1,
            data_format='channels_first'
        )
        self.decoder=Decoder(out_channels=1,
                             kernel_size=L,
                             strides=L//2)
        self.activation=ReLU()
        self.num_spks=num_spks

    def build(self,input_shape):
        super(ConvTasNet,self).build(input_shape)
    def call(self,x):
        w=self.encoder(x)
        e=self.layer_norm(w)
        e=self.bottle_neck(e)
        e=self.separation(e)
        m=self.gen_masks(e)
        m=tf.split(m,num_or_size_splits=self.num_spks,axis=1)
        m=self.activation(tf.stack(m,axis=0))
        d=[w*m[i] for i in range(self.num_spks)]
        s=[self.decoder(d[i]) for i in range(self.num_spks)]
        return tf.stack(s,axis=1)
    


if __name__=="__main__":
    model=ConvTasNet()
    x=tf.random.uniform(shape=[1,1,8000]) # eager execution to initialize the model
    y=model(x)
    print(model.summary())