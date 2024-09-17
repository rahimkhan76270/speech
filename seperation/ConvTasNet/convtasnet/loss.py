import tensorflow as tf 

@tf.function
def si_sdr_loss(original:tf.Tensor, predicted:tf.Tensor,eps:float=1e-8,loss_type:str='sisdr')->tf.Tensor:
        assert loss_type.lower() in ['sisdr','sisnr'],"loss_type should be 'sisdr' or 'sisnr'"
        original = original - tf.reduce_mean(original, axis=-1, keepdims=True)
        predicted = predicted - tf.reduce_mean(predicted, axis=-1, keepdims=True)
        dot_product = tf.reduce_sum(original * predicted, axis=-1)  # type:ignore
        original_norm_sq = tf.reduce_sum(tf.square(original), axis=-1)
        scale = dot_product / (original_norm_sq +eps)
        s_target = scale[..., tf.newaxis] * (original if loss_type == 'sisdr' else original)
        e_noise = predicted - s_target
        s_target_norm_sq = tf.reduce_sum(tf.square(s_target), axis=-1)
        e_noise_norm_sq = tf.reduce_sum(tf.square(e_noise), axis=-1)
        si_sdr = 10 * tf.math.log(s_target_norm_sq / (e_noise_norm_sq +eps)) / tf.math.log(10.0)
        return -si_sdr

@tf.function
def cdist_si_sdr(A:tf.Tensor, B:tf.Tensor,loss_type:str='sisdr')->tf.Tensor:
        assert loss_type.lower() in ['sisdr','sisnr'],"loss_type should be 'sisdr' or 'sisnr'"
        assert A.shape==B.shape ,"Tensor shape should be same"
        assert A.ndim==4 ,f"Tensor dimension should be 4 but found {A.ndim}"
        A=tf.squeeze(A,axis=2)
        B=tf.squeeze(B,axis=2)
        A_expanded = tf.expand_dims(A, axis=-2)
        B_expanded = tf.expand_dims(B, axis=-3)
        loss = si_sdr_loss(A_expanded, B_expanded,loss_type=loss_type)
        max_loss=tf.reduce_max(loss,axis=[1,2])
        return tf.reduce_mean(max_loss)