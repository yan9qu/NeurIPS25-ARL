
import numpy as np
import torch
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

__all__ = ['MAG_BERT']

def purity_score(y_true, y_pred):
        y_voted_labels = np.zeros(y_true.shape)
        labels = np.unique(y_true)
        ordered_labels = np.arange(labels.shape[0])
        for k in range(labels.shape[0]):
            y_true[y_true==labels[k]] = ordered_labels[k]
        labels = np.unique(y_true)
        bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

        for cluster in np.unique(y_pred):
            hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
            winner = np.argmax(hist)
            y_voted_labels[y_pred==cluster] = winner

        return accuracy_score(y_true, y_voted_labels)

def reinit_score(args, train_text, train_audio, train_visual, train_label, val_text, val_audio, val_visual, val_label, contribution):
    all_features = [
        train_audio, val_audio, train_visual, val_visual, train_text, val_text
    ]
    stages = ['train_audio', 'val_audio', 'train_visual', 'val_visual', 'train_text', 'val_text']
    all_purity = []

    for idx, fea in enumerate(all_features):
        print('Computing t-SNE embedding')
        result = fea

        result_2d = result.reshape(result.shape[0], -1)

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        result_2d = scaler.fit_transform(result_2d)

        
        y_pred = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(result_2d)

        if stages[idx].startswith('train'):
            purity = purity_score(train_label, y_pred)
        else:
            purity = purity_score(val_label, y_pred)
        
        all_purity.append(purity)
        print('%s purity= %.4f' % (stages[idx], purity))
        print('%%%%%%%%%%%%%%%%%%%%%%%%')

    purity_gap_audio = np.abs(all_purity[0] - all_purity[1])
    purity_gap_visual = np.abs(all_purity[2] - all_purity[3])
    purity_gap_text = np.abs(all_purity[4] - all_purity[5])

    weight_audio = torch.tanh(torch.tensor(args.move_lambda * purity_gap_audio + args.m_weight* contribution['audio']))
    weight_visual = torch.tanh(torch.tensor(args.move_lambda * purity_gap_visual + args.m_weight* contribution['video']))
    weight_text = torch.tanh(torch.tensor(args.move_lambda * purity_gap_text + args.m_weight* contribution['text']))

    print('weight audio:', weight_audio)
    print('weight visual:', weight_visual)
    print('weight text:', weight_text)

    return weight_text, weight_audio, weight_visual

def reinit(args, model, checkpoint, weight_text, weight_audio, weight_visual):
    print("Start reinit ... ")

    for name, param in model.named_parameters():
        if 'ctc_a' in name:
            init_weight = checkpoint[name]
            current_weight = param.data
            new_weight = weight_audio * init_weight + (1 - weight_audio) * current_weight
            param.data = new_weight
        elif 'ctc_v' in name:
            init_weight = checkpoint[name]
            current_weight = param.data
            new_weight = weight_visual * init_weight + (1 - weight_visual) * current_weight
            param.data = new_weight
        elif 'ctc_t' in name:
            init_weight = checkpoint[name]
            current_weight = param.data
            new_weight = weight_text * init_weight + (1 - weight_text) * current_weight
            param.data = new_weight

    return model


def get_modality_contribution(model, dataloader, device):
    model.eval()
    
    # Extract the first batch
    first_batch = next(iter(dataloader))
    
    # Extract features from the first batch
    text_feats = first_batch['text_feats'].to(device)
    audio_feats = first_batch['audio_feats'].to(device)
    video_feats = first_batch['video_feats'].to(device)
    
    batch_size = text_feats.size(0)

    modality_contributions = {'text': 0, 'audio': 0, 'video': 0}

    with torch.no_grad():
        for i in range(batch_size):
            text_sample = text_feats[i].unsqueeze(0)
            audio_sample = audio_feats[i].unsqueeze(0)
            video_sample = video_feats[i].unsqueeze(0)

            # Normal fusion output
            fused_features, _ = model(text_sample, video_sample, audio_sample)

            # Dictionary to store modality similarities
            modality_difference = {'text': 0, 'audio': 0, 'video': 0}

            # Compute contributions for each modality
            for modality in modality_difference.keys():
                # Remove one modality (set it to zero)
                modified_input = {
                    'text': torch.zeros_like(text_sample) if modality == 'text' else text_sample,
                    'audio': torch.zeros_like(audio_sample) if modality == 'audio' else audio_sample,
                    'video': torch.zeros_like(video_sample) if modality == 'video' else video_sample
                }

                # Fusion output without one modality
                modified_fused_features, _ = model(modified_input['text'], modified_input['video'], modified_input['audio'])

                # Calculate similarity (Cosine Similarity)
                similarity = nn.CosineSimilarity(dim=1)(fused_features, modified_fused_features)
                similarity = torch.abs(similarity)
                modality_difference[modality] = (1.0 - similarity).item()

            total_similarity = sum(modality_difference.values()) + 1e-8
            contribution_weights = {modality: score / total_similarity for modality, score in modality_difference.items()}

            # Accumulate contributions for the batch
            for modality in modality_contributions:
                modality_contributions[modality] += contribution_weights[modality]

    # Average the contributions across the batch
    for modality in modality_contributions:
        modality_contributions[modality] /= batch_size

    return modality_contributions


def get_feature(args, model, device, dataloader):
    model.eval()
    all_text = []
    all_audio = []
    all_visual = []
    all_label = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Feature Extraction")):
            text_feats = batch['text_feats'].to(device)
            video_feats = batch['video_feats'].to(device)
            audio_feats = batch['audio_feats'].to(device)
            label_ids = batch['label_ids'].to(device)

            _, results = model(text_feats, video_feats, audio_feats)
            
            text_feature = results['text']
            audio_feature = results['acoustic']
            visual_feature = results['visual']

            all_text.append(text_feature.cpu())
            all_audio.append(audio_feature.cpu())
            all_visual.append(visual_feature.cpu())
            all_label.append(label_ids.cpu())

    all_text = torch.cat(all_text, dim=0)
    all_audio = torch.cat(all_audio, dim=0)
    all_visual = torch.cat(all_visual, dim=0)
    all_label = torch.cat(all_label, dim=0)

    return all_text, all_audio, all_visual, all_label


def _calculate_contribution(self, text_feats, audio_feats, video_feats, yita):
    """
    Calculate average modality contributions for each batch.
    
    Returns:
        dict: Normalized contribution weights for each modality.
    """
    self.model.eval()
    batch_size = text_feats.size(0)

    modality_contributions = {'text': 0, 'audio': 0, 'video': 0}
    
    with torch.no_grad():
        for i in range(batch_size):
            text_sample = text_feats[i].unsqueeze(0)
            audio_sample = audio_feats[i].unsqueeze(0)
            video_sample = video_feats[i].unsqueeze(0)

            fused_features, _ = self.model(text_sample, video_sample, audio_sample)

            modality_difference = {'text': 0, 'audio': 0, 'video': 0}

            for modality in modality_difference.keys():
                modified_input = {
                    'text': torch.zeros_like(text_sample) if modality == 'text' else text_sample,
                    'audio': torch.zeros_like(audio_sample) if modality == 'audio' else audio_sample,
                    'video': torch.zeros_like(video_sample) if modality == 'video' else video_sample
                }

                modified_fused_features, _ = self.model(modified_input['text'], modified_input['video'], modified_input['audio'])

                similarity = nn.CosineSimilarity(dim=1)(fused_features, modified_fused_features)
                similarity = torch.abs(similarity)
                modality_difference[modality] = np.exp(yita * (1.0 - similarity).item())

            total_similarity = sum(modality_difference.values()) + 1e-8
            contribution_weights = {modality: score / total_similarity for modality, score in modality_difference.items()}

            for modality, contribution in contribution_weights.items():
                modality_contributions[modality] += contribution

    for modality in modality_contributions:
        modality_contributions[modality] /= batch_size

    return modality_contributions


    
class MAG_BERT:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        
        self.optimizer, self.scheduler = self._set_optimizer(args, data, self.model)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _set_optimizer(self, args, data, model):
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        num_train_examples = len(data.train_data_index)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    logits , _ = self.model(text_feats, video_feats, audio_feats)

                    loss = self.criterion(logits, label_ids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    
                    self.optimizer.step()
                    self.scheduler.step()
            
            outputs = self._get_outputs(args, mode = 'eval')
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
         
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def _train_both(self, args): 

        early_stopping = EarlyStopping(args)

        checkpoint = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        warmup_epochs = args.start_epoch
        flag_mask = 0
        flag_reinit = 0
        next_operation = 'resample'

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                masked_input = {
                    'text': text_feats,
                    'audio': audio_feats,
                    'video': video_feats,
                }

                if (epoch % warmup_epochs == 0) and (epoch > 0) and (next_operation == 'resample') and (flag_mask < args.mask_num):
                    contribution = _calculate_contribution(self, text_feats, audio_feats, video_feats, args.yita)
                    print(f"Contribution: {contribution}")
                    masked_input = {
                        'text': text_feats if contribution['text'] < args.contribution_threshold else torch.zeros_like(text_feats),
                        'audio': audio_feats if contribution['audio'] < args.contribution_threshold else torch.zeros_like(audio_feats),
                        'video': video_feats if contribution['video'] < args.contribution_threshold else torch.zeros_like(video_feats)
                    }

                with torch.set_grad_enabled(True):
                    self.model.train()
                    logits, _ = self.model(masked_input['text'], masked_input['video'], masked_input['audio'])
                    loss = self.criterion(logits, label_ids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    
                    self.optimizer.step()
                    self.scheduler.step()
            
            
            outputs = self._get_outputs(args, mode='eval')
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

            if (epoch % warmup_epochs == 0) and (epoch > 0):
                if next_operation == 'resample' and flag_mask < args.mask_num:
                    flag_mask += 1
                    print(f'Resample {flag_mask}')
                    next_operation = 'reinit'
                elif next_operation == 'reinit' and flag_reinit < args.reinit_num:
                    flag_reinit += 1
                    print(f'Reinit {flag_reinit}')
                    next_operation = 'resample'

                    print("Start getting training feature ...")
                    train_text, train_audio, train_visual, train_label = get_feature(args, self.model, self.device, self.train_dataloader)
                    print("Start getting evaluating feature ...")
                    val_text, val_audio, val_visual, val_label = get_feature(args, self.model, self.device, self.eval_dataloader)
                    
                    contri = get_modality_contribution(self.model, self.train_dataloader, self.device)

                    weight_text, weight_audio, weight_visual = reinit_score(
                        args,
                        train_text.numpy(),
                        train_audio.numpy(),
                        train_visual.numpy(),
                        train_label.numpy(),
                        val_text.numpy(),
                        val_audio.numpy(),
                        val_visual.numpy(),
                        val_label.numpy(),
                        contri
                    )
                    self.model = reinit(args, self.model, checkpoint, weight_text, weight_audio, weight_visual)

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)


    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                logits, _ = self.model(text_feats, video_feats, audio_feats)

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
 
                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))
                
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs

    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
        return test_results