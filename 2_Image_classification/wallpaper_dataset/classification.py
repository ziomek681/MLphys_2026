import utils


model_args = {}
model_args['batch_size'] = 64

train_loader, val_loader= utils.parse_dataset_classification('./data', 
                                                             splitratio=[.7,.3],
                                                             batchsize=model_args['batch_size'], 
                                                             normalize=False, 
                                                             downsample=True)
