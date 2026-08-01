import time
import keras
import numpy as np
from functools import partial
from src_tensorflow.chemvae import hyperparameters, mol_callbacks, train_vae

def train(config_path):

    params = hyperparameters.load_params(config_path)
    print("All params:", params)

    start_time = time.time()
    X_train, X_test = train_vae.vectorize_data(params)
    AE_only_model, encoder, decoder, kl_loss_var = train_vae.load_models(params)

    if params['optim'] == 'adam':
        optim = keras.optimizers.Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = keras.optimizers.RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = keras.optimizers.SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': train_vae.kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_callbacks.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_callbacks.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = keras.callbacks.CSVLogger(params["history_file"], append=False)
    callbacks = [ vae_anneal_callback, csv_clb]

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    xent_loss_weight = keras.backend.variable(params['xent_loss_weight'])
    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}

    AE_only_model.compile(loss=model_losses,
        loss_weights=[xent_loss_weight,
          kl_loss_var],
        optimizer=optim,
        metrics={'x_pred': ['categorical_accuracy',vae_anneal_metric]}
        )

    keras_verbose = params['verbose_print']

    AE_only_model.fit(X_train, model_train_targets,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    initial_epoch=params['prev_epochs'],
                    callbacks=callbacks,
                    verbose=keras_verbose,
                    validation_data=[ X_test, model_test_targets]
                    )

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

if __name__ == '__main__':

    config_path = r'src_tensorflow/experiments/exp.json'
    train(config_path)