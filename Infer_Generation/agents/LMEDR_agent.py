import random
import logging
from pprint import pformat
from typing import List, Dict
from collections import defaultdict
from functools import partial
from tqdm.auto import tqdm
import torch
from parlai.core.agents import Agent
from parlai.scripts.eval_model import setup_args as base_setup_args
from transformers import BartTokenizer
from agents.lmedr_model.modeling_bart import LMEDRModel
from agents.lmedr_model.eval_utils import create_encoder_input, create_decoder_input, pad_dataset

class LMEDResponseAgent(object):
    def __init__(self):
        # load the model
        checkpoint_path = 'agents/lmedr_model/checkpoints'
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
        self.model = LMEDRModel.from_pretrained(checkpoint_path)
        self.query_id, self.res_id, self.latent_id, self.persona_id, self.partner_id = \
            self.tokenizer.convert_tokens_to_ids(['<query>', '<response>', '<latent>', '<persona>', '<partner>'
            ])
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        self.turn_id = 1

        print("Model loaded!!")
        
        self.max_input_tokens = 1024
        
        
    def tokenize_conversation(self, conversation):
        def tokenize(text):
            return self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(text.strip(), add_prefix_space=True)
            )
       
        persona = [tokenize(line.strip()) for line in conversation['persona B']]
        # partner = [tokenize(line.strip()) for line in conversation['persona A']]
        partner = [] # Baseline not trained with the partner personaj
        history = [tokenize(line['text'].strip()) for line in conversation['dialogue']]
        return persona, partner, history
   
    def prepare_tensors(self, conversation):
        persona, partner, history = self.tokenize_conversation(conversation)
        input_ids, attention_mask, per_input_ids, per_attention_mask = create_encoder_input(persona,
            # partner,
            history,
            self.query_id, 
            self.res_id, 
            self.latent_id,
            self.persona_id, 
            # self.partner_id,
            self.sep_id, 
            self.eos_id
        )
        tensor_input_ids = torch.tensor(input_ids, device=self.device)[-self.max_input_tokens:].unsqueeze(0)
        tensor_attention_mask = torch.tensor(attention_mask, device=self.device)[-self.max_input_tokens:].unsqueeze(0)
        tensor_per_input_ids = torch.tensor(per_input_ids, device=self.device)[-self.max_input_tokens:].unsqueeze(0)
        tensor_per_attention_mask = torch.tensor(per_attention_mask, device=self.device)[-self.max_input_tokens:].unsqueeze(0)
        return tensor_input_ids, tensor_attention_mask, tensor_per_input_ids, tensor_per_attention_mask

        
        
        
    def generate_responses(self, test_data: List[Dict], api_responses: List[str]) -> List[str]:
        
        all_responses = []
        
        for conversation in tqdm(test_data):
            tensor_input_ids, tensor_attention_mask,tensor_per_input_ids, tensor_per_attention_mask= self.prepare_tensors(conversation)
            with torch.no_grad():
                out_ids = self.model.generate(
                    input_ids=tensor_input_ids,
                    attention_mask=tensor_attention_mask,
                    per_input_ids=tensor_per_input_ids,
                    per_attention_mask=tensor_per_attention_mask,
                    max_length=50,
                    num_beams=2
                )
                
            out_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
            response = out_text[0].strip()
            all_responses.append(response)
            
        self.turn_id = self.turn_id % 7 + 1
        
        response = {
        "use_api": False,                                    # Cannot use API if GPU true is set in aicrowd.json
        "prompts": ["" for _ in test_data],                  # Cannot use API if GPU true is set in aicrowd.json
        "max_generated_tokens": [0 for _ in test_data],
        "final_responses": all_responses
    }
        print(response)

        
        return response
              