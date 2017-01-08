# -*- coding: utf-8 -*-
def get_nnet_configuration():
    return {
               'train_size': -1,
               'valid_size': -1,
                'test_size': -1,
          'valid_frequency': 500,
           'save_frequency': 500,
                   'epochs': 50,
              'random_seed': 1484,
        'hidden_layer_size': 2048,
                'frequency': 44100,
                   'borrow': True,
                  'dropout': False,
             'reload_model': False,
             'shuffle_data': False,
         'shuffle_bin_data': False,
          'processing_unit': 'default',
        'exception_verbose': 'high',
              'dataset_dir': 'data/',
               'model_name': 'nnerd_model',
                'model_dir': 'models/',
                     'mode': 'straight',
        #TODO:
           'multithreading': False,
                 'channels': 2,
            'compress_step': 0,
    }
