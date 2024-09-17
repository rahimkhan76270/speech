from convtasnet.loss import cdist_si_sdr 
from convtasnet.model import ConvTasNet
from convtasnet.dataset import ConvTasNetDataGenerator
import tensorflow as tf
from tensorflow.keras import optimizers # type:ignore
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard # type:ignore
from tensorflow.data import Dataset #type:ignore
import yaml


if __name__=="__main__":
    with open('./config.yaml','r') as file:
        config=yaml.safe_load(file)
    
    # model parameters
    model_params=config.get('model')
    N=model_params.get("N")
    L=model_params.get("L")
    B=model_params.get("B")
    H=model_params.get("H")
    P=model_params.get("P")
    X=model_params.get("X")
    R=model_params.get("R")
    num_spks=model_params.get("num_spks")
    causal=model_params.get('causal')

    # dataset parameters
    dataset_params=config.get('dataset')
    train_audio_path=dataset_params.get('train_audio_path')
    test_audio_path=dataset_params.get('test_audio_path')
    val_audio_path=dataset_params.get('val_audio_path')
    sample_rate=dataset_params.get('sample_rate')
    audio_chunk_len=dataset_params.get('audio_chunk_len')
    file_ext=dataset_params.get('file_ext')
    batch_size=dataset_params.get('batch_size')
    train_data_len=dataset_params.get('train_data_len')
    test_data_len=dataset_params.get('test_data_len')
    val_data_len=dataset_params.get('val_data_len')

    # optimizer 
    optimizer_params=config.get('optimizer_config')
    optimizer_name=optimizer_params.get("optimizer_name")
    learning_rate=optimizer_params.get('learning_rate')
    weight_decay=optimizer_params.get('weight_decay')
    optimizers_list={
        'adam':optimizers.Adam(learning_rate=learning_rate,weight_decay=weight_decay),
        'sgd':optimizers.SGD(learning_rate=learning_rate,weight_decay=weight_decay),
        'adamw':optimizers.AdamW(learning_rate=learning_rate,weight_decay=weight_decay)
    }
    # loss
    loss_type=config.get('loss_type')

    train_config=config.get('training')
    num_epochs=train_config.get('num_epochs')
    steps_per_epoch=train_config.get('steps_per_epoch')
    verbose=train_config.get('verbose')
    # early stopping config
    es_config=config.get('early_stopping_config')
    es_monitor=es_config.get('monitor')
    es_min_delta=es_config.get('min_delta')
    es_patience=es_config.get('patience')

    # model checkpoint config
    model_checkpoint_config=config.get("model_checkpoint_config")
    model_checkpoint_filepath=model_checkpoint_config.get('filepath')
    model_checkpoint_monitor=model_checkpoint_config.get('monitor')
    model_checkpoint_save_best=model_checkpoint_config.get('save_best_only')
    model_checkpoint_save_frequency=model_checkpoint_config.get('save_freq')

    # reduce lr on plateau
    lr_plateau=config.get('reduce_lr_on_plateau')
    lr_plateau_monitor=lr_plateau.get('monitor')
    lr_plateau_factor=lr_plateau.get('factor')
    lr_plateau_patience=lr_plateau.get('patience')
    lr_plateau_min_delta=lr_plateau.get('min_delta')
    lr_plateau_cooldown=lr_plateau.get('cooldown')
    lr_plateau_min_lr=lr_plateau.get('min_lr')
    
    # tensorboard config
    tensorboard_config=config.get('tensorboard_config')
    log_dir=tensorboard_config.get('log_dir')
    write_graph=tensorboard_config.get('write_graph')
    
    # data generator
    train_data_generator=ConvTasNetDataGenerator(
        folder_path=train_audio_path,
        audio_chunk_len=audio_chunk_len,
        num_spks=num_spks,
        sample_rate=sample_rate,
        data_len=train_data_len,
        file_ext=file_ext
    )

    test_data_generator=ConvTasNetDataGenerator(
        folder_path=test_audio_path,
        audio_chunk_len=audio_chunk_len,
        num_spks=num_spks,
        sample_rate=sample_rate,
        data_len=test_data_len,
        file_ext=file_ext
    )

    val_data_generator=ConvTasNetDataGenerator(
        folder_path=val_audio_path,
        audio_chunk_len=audio_chunk_len,
        num_spks=num_spks,
        sample_rate=sample_rate,
        data_len=val_data_len,
        file_ext=file_ext
    )

    train_dataset=Dataset.from_generator(train_data_generator.generator,
                                     output_signature=(
                                         tf.TensorSpec(shape=(1,sample_rate*audio_chunk_len),dtype='float32'),  # type:ignore
                                         tf.TensorSpec(shape=(2,1,sample_rate*audio_chunk_len),dtype='float32')  # type:ignore
                                     ))
    test_dataset=Dataset.from_generator(test_data_generator.generator,
                                        output_signature=(
                                            tf.TensorSpec(shape=(1,sample_rate*audio_chunk_len),dtype='float32'), # type:ignore
                                            tf.TensorSpec(shape=(2,1,sample_rate*audio_chunk_len),dtype='float32') # type:ignore
                                        ))
    val_dataset=Dataset.from_generator(val_data_generator.generator,
                                        output_signature=(
                                            tf.TensorSpec(shape=(1,sample_rate*audio_chunk_len),dtype='float32'),  # type:ignore
                                            tf.TensorSpec(shape=(2,1,sample_rate*audio_chunk_len),dtype='float32') # type:ignore
                                        ))
    
    train_data_loader=train_dataset.batch(batch_size)
    test_data_loader=test_dataset.batch(batch_size)
    val_data_loader=val_dataset.batch(batch_size)

    # callbacks setup
    # early stopping
    early_stopping_callback=EarlyStopping(
        monitor=es_monitor,
        min_delta=es_min_delta,
        patience=es_patience
    )
    # model_checkpoints
    model_checkpoint_callback=ModelCheckpoint(
        filepath=model_checkpoint_filepath,
        monitor=model_checkpoint_monitor,
        save_best_only=model_checkpoint_save_best,
        save_freq=model_checkpoint_save_frequency
    )
    # reduce lr on plateau
    lr_callback=ReduceLROnPlateau(
        monitor=lr_plateau_monitor,
        factor=lr_plateau_factor,
        patience=lr_plateau_patience,
        min_delta=lr_plateau_min_delta,
        cooldown=lr_plateau_cooldown,
        min_lr=lr_plateau_min_lr
    )

    # tensorboard callback
    tensorboard_callback=TensorBoard(
        log_dir=log_dir,
        write_graph=write_graph
    )

    optimizer=optimizers_list[optimizer_name]
    model=ConvTasNet(
        N=N,
        L=L,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
        num_spks=num_spks,
        causal=causal
    )
    model.build(input_shape=(None,1,sample_rate*audio_chunk_len))
    # quick checkup
    x=tf.random.uniform(shape=[1,1,sample_rate*audio_chunk_len])
    y=model(x)
    print(model.summary())
    model.compile(optimizer=optimizer,loss=cdist_si_sdr)
    print('fitting the model')
    model.fit(
        train_data_loader,
        epochs=num_epochs,
        validation_data=val_data_loader,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping_callback,model_checkpoint_callback,lr_callback,tensorboard_callback]
    )
    print("model training completed")

    test_results=model.evaluate(
        train_data_loader,
        verbose=verbose,
        steps=steps_per_epoch,
        callbacks=[tensorboard_callback],
        return_dict=True
    )
    print(f"test results : {test_results}")
    print(f"all the model weights are saved at {model_checkpoint_filepath}")