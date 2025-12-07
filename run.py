import os
import random
from configs.base import ParamManager
from data.base import DataManager
from methods import method_map
from backbones.base import ModelManager
from utils.functions import set_torch_seed, save_results, set_output_path
import argparse
import logging
import datetime
import itertools
import warnings

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    # Basic settings
    parser.add_argument('--logger_name', type=str, default='mag_bert', help="Logger name for multimodal intent analysis.")
    
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
    
    parser.add_argument('--gpu_id', type=str, default='1', help="GPU index to use.")
    
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading.")
    
    # Dataset and method configuration
    parser.add_argument('--dataset', type=str, default='MOSI', help="Name of the dataset to use.")

    parser.add_argument('--data_mode', type=str, default='multi-class', help="Task mode (multi-class or binary-class).")

    parser.add_argument('--method', type=str, default='mag_bert', help="Method to use (text, mult, misa, mag_bert).")

    parser.add_argument("--text_backbone", type=str, default='bert-base-uncased', help="Backbone model for the text modality.")
    
    parser.add_argument("--n_classes", type=int, default=7, help="Number of label classes.")
    
    # Data paths
    parser.add_argument("--data_path", default='/data', type=str,
                        help="Input data directory containing text, video, and audio data.")

    parser.add_argument('--video_data_path', type=str, default='video_data', help="Directory of video data.")

    parser.add_argument('--audio_data_path', type=str, default='audio_data', help="Directory of audio data.")

    parser.add_argument('--video_feats_path', type=str, default='video_feats.pkl', help="Path to video features file.")

    parser.add_argument('--audio_feats_path', type=str, default='audio_feats.pkl', help="Path to audio features file.")
    
    # Output paths
    parser.add_argument('--log_path', type=str, default='logs', help="Directory for log files.")
    
    parser.add_argument('--cache_path', type=str, default='cache', help="Caching directory for pre-trained models.")
    
    parser.add_argument('--results_path', type=str, default='results', help="Directory to save results.")

    parser.add_argument("--output_path", default='outputs', type=str, 
                        help="Output directory where all training data will be written.") 

    parser.add_argument("--model_path", default='models', type=str, 
                        help="Output directory for model predictions and checkpoints.") 

    parser.add_argument("--config_file_name", type=str, default='mag_bert', help="Name of the configuration file.")

    parser.add_argument("--results_file_name", type=str, default='mag_mosi_both.csv', help="Filename for experimental results.")
    
    # Training settings
    parser.add_argument("--train", default=True, help="Whether to train the model.")

    parser.add_argument("--tune", action="store_true", help="Whether to tune hyperparameters.")

    parser.add_argument("--save_model", action="store_true", help="Whether to save the trained model.")

    parser.add_argument("--save_results", default=True, help="Whether to save final results.")
    
    parser.add_argument('--pattern', type=str, default='both', help="Training pattern (normal or both).")
    
    parser.add_argument("--start_epoch", type=int, default=2, help="Epoch to start applying both operations.") 
    
    # Resample parameters
    parser.add_argument("--mask_start_epoch", type=int, default=2, help="Epoch to start masking modalities.")    

    parser.add_argument("--contribution_threshold", type=float, default=0.5, help="Threshold for modality contribution.")   

    parser.add_argument("--mask_num", type=int, default=1, help="Number of resample operations to perform.") 

    parser.add_argument("--yita", type=float, default=10, help="Eta (η) parameter for contribution calculation.")  
    
    # Reinit parameters
    parser.add_argument("--reinit_epoch", type=int, default=2, help="Epoch to start reinitialization.") 

    parser.add_argument("--reinit_num", type=int, default=1, help="Number of reinitialization operations to perform.")  

    parser.add_argument("--move_lambda", type=int, default=3, help="Lambda parameter for weight calculation in reinit.")

    parser.add_argument("--m_weight", type=float, default=0.01, help="Weight for modality contribution in reinit.") 

    args = parser.parse_args()

    return args

def set_logger(args):
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name =  f"{args.method}_{args.dataset}_{args.data_mode}_{time}"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.logger_name + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger

def run(args, debug_args=None):
    
    logger = set_logger(args)
    args.pred_output_path, args.model_output_path = set_output_path(args)
    
    set_torch_seed(args.seed)

    logger.info("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.info(f"{k}: {args[k]}")
    logger.info("="*30+" End Params "+"="*30)
    
    data = DataManager(args)
    method_manager = method_map[args.method]
    
    if args.method == 'text':
        method = method_manager(args, data)
    else:
        model = ModelManager(args)
        method = method_manager(args, data, model)
        
    logger.info('Multimodal intent recognition begins...')

    if args.train:
        if args.pattern == 'normal':
            logger.info('method._train')
            method._train(args)
        elif args.pattern == 'both':
            logger.info('method._train_both')
            method._train_both(args)
        logger.info('Training is finished...')

    logger.info('Testing begins...')
    outputs = method._test(args)
    logger.info('Testing is finished...')
    
    logger.info('Multimodal intent recognition is finished...')

    if args.save_results:
        
        logger.info('Results are saved in %s', str(os.path.join(args.results_path, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)
    
    
    
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    args = parse_arguments()

    datasets = ['MOSI']
    seeds = [random.randint(0, 10000)]
    
    param = ParamManager(args)
    args = param.args

    if args.tune:
        debug_args = {}
        
        for k, v in args.items():
            if isinstance(v, list):
                debug_args[k] = v
        
        for result in itertools.product(*debug_args.values()):
            for i, key in enumerate(debug_args.keys()):
                args[key] = result[i]         
            
            for dataset in datasets:
                for seed in seeds:
                    args.dataset = dataset
                    args.seed = seed
                    run(args, debug_args=debug_args)

    else:
        for dataset in datasets:
            for seed in seeds:
                args.dataset = dataset
                args.seed = seed
                run(args)

