from voxseg_tf.model import CNNBiLSTMModel
from voxseg_tf.dataset import DataGenerator  # type:ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type:ignore
import tensorflow as tf
from tensorflow.data import Dataset  # type:ignore


if __name__=='__main__':
    # data folder should contain individual speaker files in separate folder 
    # folder:
    # ->spk1:
    # ->->file1.wav
    # ->->file2.wav
    # ->spk2:
    # ->->file1.wav
    # ->->file2.wav   and so on
    # every file should have 16000 sample rate if possible
    train_data_folder="/kaggle/input/kathbath-wav-hindi/valid/wav/train"
    test_data_folder="/kaggle/input/kathbath-wav-hindi/valid/wav/test"
    val_data_folder="/kaggle/input/kathbath-wav-hindi/valid/wav/val"

    train_data_function=DataGenerator('/kaggle/input/kathbath-wav-hindi/valid/wav/train')
    train_data_generator=Dataset.from_generator(train_data_function.generator,output_signature=(tf.TensorSpec(shape=(6,32,32,1),dtype='float'),tf.TensorSpec(shape=(6,2),dtype='float')))  # type:ignore   
    train_dataset=train_data_generator.batch(50).repeat() 

    val_data_function=DataGenerator('/kaggle/input/kathbath-wav-hindi/valid/wav/val',data_len=1000)
    val_data_generator=Dataset.from_generator(val_data_function.generator,output_signature=(tf.TensorSpec(shape=(6,32,32,1),dtype='float'),tf.TensorSpec(shape=(6,2),dtype='float'))) # type:ignore
    val_dataset=val_data_generator.batch(50).repeat()

    test_data_function=DataGenerator('/kaggle/input/kathbath-wav-hindi/valid/wav/test',data_len=1000)
    test_data_generator=Dataset.from_generator(val_data_function.generator,output_signature=(tf.TensorSpec(shape=(6,32,32,1),dtype='float'),tf.TensorSpec(shape=(6,2),dtype='float'))) # type:ignore
    test_dataset=test_data_generator.batch(50).repeat()

    # make sure model_checkpoints directory exists
    model_checkpoint_callback=ModelCheckpoint(
    filepath='/kaggle/working/model_checkpoints/{epoch:02d}-{val_loss:.2f}.keras',
                                 save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True)
    
    model=CNNBiLSTMModel()
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),   #type:ignore
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    model.fit(train_dataset,epochs=1,steps_per_epoch=200,validation_data=val_dataset,
      validation_steps=20,
      callbacks=[model_checkpoint_callback],verbose=1)
    
    result=model.evaluate(
        test_dataset,steps=20
    )
    print(result)