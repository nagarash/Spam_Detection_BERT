import os
import torch
import pickle
from transformers import BertTokenizerFast


class TextTokenize:
    
    def __init__(self):

        # Load the BERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def token_and_encode(self, text, label, pad_length, save_dir, save_label):
        """
        tokenize and encode text sequences padding/truncating based on pad_length
        returns input_id and att_mask tensors
        """
    
        tokens = self.tokenizer.batch_encode_plus(
            text.tolist(),
            max_length = pad_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            return_attention_mask = True  # Generate the attention mask
        )
        
        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        pt_y = torch.tensor(label.tolist())
        
        # save tensors
        os.makedirs(save_dir, exist_ok=True)
        with open("{}/{}_tensors.pkl".format(save_dir, save_label), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([seq, mask, pt_y], f)
        