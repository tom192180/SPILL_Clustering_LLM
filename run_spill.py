import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer,set_seed
from tqdm import tqdm
import csv
import json
import random
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch import cuda
import argparse
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import time
from datasets import load_dataset
import re
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics import cluster


 

class GetEmbeddings:
    
    def __init__(self, 
                 dataset_name, 
                 metric,
                 mode,
                 first_selection_num,
                 top_conf,
                 llm_for_encoder,
                 model_name,
                 sampling = 'no',
                 pooling = 'mean', 
                 seed_value = 42):
        allowed_datasets = ['sgd','clinc150','bank77','mtop','massive']
        assert dataset_name in allowed_datasets, f"dataset_name must be one of {allowed_datasets}"
        self.model_name = model_name
        self.mode = mode
        self.sampling = sampling
        self.dataset_name = dataset_name
        self.seed_value = seed_value
        self.intent_list = None
        self.ut_list = None
        self.ut_intent_map = dict()
        self.intent_ut_map = dict()
        self.cloest_ut_list = None
        self.cloest_ut_list_llm = None
        self.cloest_ut_list_gd = None
        self.ratio = 999
        self.llm_gen_num = 999
        self.metric = metric
        self.n = first_selection_num
        self.llm_for_encoder = llm_for_encoder
        self.top_conf = top_conf
        self.mistake_count = 999
        self.gd_num = 999
        print(f'self.n: {self.n }; self.metric:{self.metric};self.sampling:{self.sampling}')
        if 'sentence-transformers' in self.model_name or 'e5-large' in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name,
                                                     device_map='auto')
        elif 'instructor' in self.model_name:
            self.model = INSTRUCTOR(self.model_name, device='cuda')

            
        else:
            torch_dtype = torch.float16 if 'gemma-2-9b-it' in self.model_name else torch.bfloat16
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                           token= 'hf_ID',
                                                           padding_side="left")
            self.model  = AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype=torch_dtype,
                                                           token= 'hf_ID',
                                                           device_map='auto')
            self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id 
            
            
         

        

        self.embeddings = None
        self.pooling = pooling
        
    def load_data(self, filepath):
        random.seed(self.seed_value)

        if self.dataset_name == 'bank77':
            ds = load_dataset("legacy-datasets/banking77")
            list_of_uts = [example['text'] for example in ds['test']]
            list_of_intent = [example['label'] for example in ds['test']]
            
        else:
            if self.dataset_name ==  'sgd':
                print(f'seed_value: {self.seed_value}')
                filepath = '..'

            elif self.dataset_name ==  'clinc150':
                filepath = '..'
            elif self.dataset_name ==  'mtop':
                filepath = '..'
            elif self.dataset_name ==  'massive':
                filepath = '..'
            list_of_uts = []
            list_of_intent = []
            with open(filepath, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                
                # Read each row from the CSV file and append it to the list_of_lists
                for row in csvreader:
                    ut = row[0]
                    intent = row[1]
                    list_of_uts.append(ut)
                    list_of_intent.append(intent)
             
        combined = list(zip(list_of_uts, list_of_intent))
        random.shuffle(combined)
        shuffled_uts, shuffled_intent = zip(*combined)
        self.intent_list =  shuffled_intent
        self.ut_list = shuffled_uts
        self.ut_intent_map = {ut:intent for ut, intent in zip(self.ut_list,self.intent_list)}
        for ut, intent in self.ut_intent_map.items():
            if intent not in self.intent_ut_map:
                self.intent_ut_map[intent] = [ut]
            else:
                self.intent_ut_map[intent].append(ut)        
        print(f'There are {len(set(self.intent_list))} unique intents.')
        print(f'There are {len(self.ut_list)} examples.')


    def get_distance_matrix(self, input_eb):
        if self.metric == 'euclidean':
            distance_matrix = euclidean_distances(input_eb)
        elif self.metric == 'cosine':
            similarity_matrix = cosine_similarity(input_eb)
            distance_matrix = 1 - similarity_matrix        
        np.fill_diagonal(distance_matrix, float('inf'))   # Exclude the sentence itself by setting its distance to a very high value

        return distance_matrix

    def find_closest_sentences_gd(self,input_collection):
        random.seed(self.seed_value)
        if '_llm' not in self.mode:
            self.mode += '_gd'
        else:
            self.mode = self.mode.replace('_llm', '_gd')
        # To store the results
        closest_pairs = []
        closest_pairs_ebd = []

        closest_pairs_rd = []
        closest_pairs_ebd_rd = []

        count = 0
        for ut_list in self.cloest_ut_list:
            first_ut = ut_list[0]
            intent = self.ut_intent_map[first_ut]

            correct_ut_list = [ut for ut in ut_list[1:] if self.ut_intent_map[ut] == intent]
            correct_ut_list_ebd = [self.initial_emb[ut] for ut in ut_list[1:] if self.ut_intent_map[ut] == intent]
            count += len(correct_ut_list)/len(self.cloest_ut_list)
             
            # Ordered
            
            correct_ut_list_ebd_odr = [self.initial_emb[first_ut]] + correct_ut_list_ebd
            correct_ut_list_odr = [first_ut] + correct_ut_list
                        
            closest_pairs.append(correct_ut_list_odr)
            average_embedding_odr = np.mean(np.array(correct_ut_list_ebd_odr), axis=0)
            closest_pairs_ebd.append(average_embedding_odr)                
        
        self.cloest_ut_list_gd = closest_pairs         
        
                
        # Odr
        tmp = self.mode
        self.mode += '_DrP'
        self.embeddings = np.array(closest_pairs_ebd)
        self.gd_num = count
        self.evaluate()
        self.mode = tmp
        self.embeddings = None
        print(f'Averge gd number:{round(count,2)}')
    def find_closest_sentences(self,input_collection):
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)
        self.initial_emb = {ut:ebd for ut, ebd in zip(input_collection,self.embeddings)} 
        if self.metric == 'cosine' and 'cs' not in self.mode:
            if self.sampling == 'yes':
                self.mode += f'_cs_lsws_sp'

            else:
                self.mode += '_cs_lsws'
        else:
            if self.sampling == 'yes':
                self.mode += f'_lsws_sp'

            else:
                self.mode += '_lsws'            
        distance_matrix = self.get_distance_matrix(input_eb = self.embeddings)
        # To store the results
        closest_pairs = []
        closest_pairs_ebd = []
        closest_pairs_rd = [] # to evaluate Rd list
        closest_pairs_ebd_rd = [] # to evaluate Rd embeddings
        # Iterate over each sentence and its embedding
        for i, sent in enumerate(input_collection):
            distances = distance_matrix[i]
            ##Find the indices of the top n closest sentences
            if self.sampling == 'yes':
                
                top_ids  = np.argsort(distances)[:self.top_conf]

                rd_ids = []
                sorted_ids = np.argsort(distances)[self.top_conf:]
                shuffled_elements = np.copy(sorted_ids)
                np.random.shuffle(shuffled_elements)
                rd_num = self.n - self.top_conf
                if rd_num != 0:
                    random_ids_chunks = np.array_split(shuffled_elements, rd_num)
                    for ids_chunk in random_ids_chunks:
                        ids_chunk = ids_chunk.tolist()
                        tmp = sorted(ids_chunk, key=lambda x: np.where(sorted_ids == x)[0][0])[0]
                        rd_ids.append(tmp)
                    
                select_n_ids = list(top_ids) + rd_ids


            elif self.sampling == 'no':
                select_n_ids = np.argsort(distances)[:self.n]
            close_sent_list = [input_collection[select_n_ids[i]] for i in range(len(select_n_ids))]
            close_sent_list_ebd = [self.embeddings[select_n_ids[i]] for i in range(len(select_n_ids))]
            sent_ebd = self.embeddings[i] 
            
            # Get closest neighbor pooling result (random)
            
            combine = list(zip(close_sent_list, close_sent_list_ebd))  # Convert to a list first
            sampled_pairs = random.sample(combine, len(combine))  # Sample pairs
            close_sent_list_rd, close_sent_list_ebd_rd = map(list, zip(*sampled_pairs))                  
            tmp_rd = [sent] + close_sent_list_rd
            close_sent_list_ebd_rd = [sent_ebd] + close_sent_list_ebd_rd   
            
            closest_pairs_rd.append(tmp_rd) 
            average_embedding_rd = np.mean(np.array(close_sent_list_ebd_rd), axis=0) 
            closest_pairs_ebd_rd.append(average_embedding_rd)            
            


            # DSC similiarity
            tmp_dsc = [sent] + close_sent_list 
            close_sent_list_ebd_dsc = [sent_ebd] + close_sent_list_ebd 
            
            closest_pairs.append(tmp_dsc) 
            average_embedding_dsc = np.mean(np.array(close_sent_list_ebd_dsc[:11]), axis=0) 
            closest_pairs_ebd.append(average_embedding_dsc)            
 
            


        self.cloest_ut_list = closest_pairs     
            
        # Evaluate Rd    
        self.ratio = self.get_correct_ratio(closest_pairs_rd)     
        tmp = self.mode    
        self.mode += '_RdDrP'
        self.embeddings = np.array(closest_pairs_ebd_rd)
        self.evaluate()
        self.mode = tmp
        
        # Evaluate Odr; 
        # DSC or ASC will have the same result as odr beacuse we just do mean pooling
        # But the closest_pairs for both will be different and will be use differently for 2nd stage   

        self.ratio = self.get_correct_ratio([ut_list[:11] for ut_list in closest_pairs])  
        tmp = self.mode
        self.mode += '_OdrDrP'
        self.embeddings = np.array(closest_pairs_ebd)
        self.evaluate()
        self.mode = tmp
        self.embeddings = None
        first_length = len(self.cloest_ut_list[0])
        self.ratio = 999
        print(f'The lenth for cloest_ut_list are same: {all(len(sublist) == first_length for sublist in self.cloest_ut_list)}; len: {first_length}')

    def add_prompt(self, list_of_list):
        result_ut = []
        for ut_list in list_of_list:
            target_ut = ut_list[0]
            cand_ut = '\n'.join([f'{i}. {ut}' for i, ut in enumerate(ut_list[1:], start=1)]) 
            prompt_bg = (
                "Task Instructions:\n\n"
                
                "Step 1: Identify Intent Clusters\n"
                "Review the Candidate Utterances to identify their individual intents and group them into clusters based on shared intent. Candidates may either align with the same cluster as the Target Utterance or belong to entirely different clusters.\n"                 
                "Note: Intent refers to the request or the purpose the user wants to achieve.\n\n"
               
                "Step 2: Match Intent with Target Utterance\n"
                "Compare each Candidate's intent to the Target Utterance, using the clusters you identified. Select only Candidates from the same intent cluster as the Target Utterance.\n"
                "Note: Choose a Candidate only if its intent clearly aligns with the Target Utterance's purpose.\n\n"
                
                "Answer Format:\n"
                "Only provide the final selection of Candidate Utterances by listing their numbers if they match the Target Utterance intent or request.\n"
                "1. If Candidates 3, 4, 9, and 11 match, write: The Candidate utterances numbers are: 3, 4, 9, 11\n"
                "2. If no Candidate matches, write: The Candidate utterances numbers are: none\n"
                "Note: Stick to the answer format and avoid providing extra explanations.\n\n"
                
                "Task:\n"
                f"Target Utterance: {target_ut}\n"
                f"Candidate Utterances:\n{cand_ut}\n"
            )
            result_ut.append(prompt_bg)
        return result_ut

    def get_correct_ratio(self, list_of_list):
        ratio = 0
        correct_ut_num = 0
        tt_len = sum([len(ele) - 1 for ele in list_of_list])
        for ut_list in list_of_list:
            first_ut = ut_list[0]
            intent = self.ut_intent_map[first_ut]
            correct_ut_list_tmp =[1 if self.ut_intent_map[ut] == intent else 0 for ut in ut_list[1:]]
            ratio += sum(correct_ut_list_tmp)/tt_len
            
        return ratio      
    

    def find_closest_sentences_llm(self):
        random.seed(self.seed_value)
        shuffled_list = [[ut_list[0]] + random.sample(ut_list[1:], len(ut_list) - 1) for ut_list in self.cloest_ut_list] 
        if 'sentence-transformers' not in self.model_name and 'instructor' not in self.model_name and 'e5-large' not in self.model_name: 
            self.tokenizer.padding_side = 'left' 

        self.mode += '_llmCoT' if '_gd' not in self.mode else self.mode.replace('_gd', '_llmCoT')

        if self.model is None: # this is for Echo
            torch_dtype = torch.float16 if 'gemma-2-9b-it' in self.model_name else torch.bfloat16
            self.model  = AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype=torch_dtype,
                                                           token= 'hf_ID',
                                                           device_map='auto')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                           token= 'hf_ID',
                                                           padding_side="left")
            self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id 
                        
                       
        if 'sentence-transformers' in self.model_name or 'instructor' in self.model_name or 'e5-large' in self.model_name: 
            print(f'llm_for_encoder: {self.llm_for_encoder}')
            if 'gemma-2-9b-it' in self.llm_for_encoder:
                self.mode += 'gemmaleft'
            elif 'Qwen2.5-7B-Instruct' in self.llm_for_encoder:
                self.mode += 'Qwenleft'
            elif 'llama' in self.llm_for_encoder:
                self.mode += 'llamaleft'
            elif 'Ministral' in self.llm_for_encoder:
                self.mode += 'mstlleft'
                
                

            del self.model
            torch.cuda.empty_cache()
             
             
            torch_dtype = torch.float16 if 'gemma-2-9b-it' in self.llm_for_encoder else torch.bfloat16

            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_for_encoder,
                                                           token= 'hf_ID',
                                                           padding_side="left")
            self.model  = AutoModelForCausalLM.from_pretrained(self.llm_for_encoder,torch_dtype=torch_dtype,
                                                           token= 'hf_ID',
                                                           device_map='auto')

            self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id 
            

                         
            
        closet_pairs_pt = self.add_prompt(shuffled_list)
        print(f'length of closet_pairs_llm is: {len(closet_pairs_pt)}')
        sep_words = [] # to get prompt length
        closet_pairs_pt_add_tmp = []
        for chunk in closet_pairs_pt:
            if 'llama' in self.llm_for_encoder or 'Qwen' in self.llm_for_encoder:
                sys_instruct = "You are a chatbot that always answers with accurate responses"    
                text_format = [{"role": "system","content": sys_instruct,},{"role": "user", "content": chunk}] 
            elif 'gemma' in self.llm_for_encoder or 'Ministral' in self.llm_for_encoder or 'Phi' in self.llm_for_encoder:
                text_format = [{"role": "user", "content": chunk}]
            text_with_cht_tmp = self.tokenizer.apply_chat_template(text_format, add_generation_prompt=True, tokenize=False)
            sp = text_with_cht_tmp[-200:]
            
            closet_pairs_pt_add_tmp.append(text_with_cht_tmp)
            sep_words.append(sp)  
        chunk_length = [len(ut_list) for ut_list in shuffled_list]    
        batch_size = 10
        closest_pair_id = []
        mistake_count = 0
        count_generate_id_ratio = 0
        for i in range(0, len(closet_pairs_pt_add_tmp), batch_size):
            batch_chunk_length = chunk_length[i:i + batch_size]
            batch_sep_words_tk = self.tokenizer(sep_words[i:i + batch_size], padding=True,  return_tensors="pt") # to make sure the format is the same as input en-de 
            batch_texts = closet_pairs_pt_add_tmp[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True,  return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            outputs = self.model.generate(**inputs,max_new_tokens = 100)    
        
            # Decode generated tokens to text
            batch_generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_sep_words = self.tokenizer.batch_decode(batch_sep_words_tk['input_ids'], skip_special_tokens=True) # to make sure the format is the same as input en-de 

            for pred_text , sep_word, ck_length in zip(batch_generated_texts, batch_sep_words,batch_chunk_length):
                mistake_tmp = False
                pred_split = pred_text.split(sep_word)
                pred = pred_split[-1].lower()
                    
                tmp = []
                if 'none' in pred:
                    tmp = tmp
                else:
                    numbers = [int(num) for num in re.findall(r'\d+', pred)] 
                    for ind in numbers:
                        if ind >= ck_length:
                            mistake_tmp = True
                        else:
                            tmp.append(ind)
                        

                
                tmp = [0] + random.sample(tmp, len(tmp))
              
                if mistake_tmp:                     
                    mistake_count += 1
                    tmp = [0]
                    with open(f"../lsws_{self.mode}_{self.dataset_name}_{mistake_count}_{self.seed_value}sd.txt", "w") as file:
                        file.write(pred_text)
                  
                
                count_generate_id_ratio += (len(tmp) - 1)/len(self.ut_list)    
                closest_pair_id.append(tmp)
                                
        self.llm_gen_num = count_generate_id_ratio
        print(f'**Total Mistake**:{mistake_count}')
        self.mistake_count= mistake_count
        print(f'**LLM select uts num**:{round(count_generate_id_ratio,1)}')

        #  original
        llm_closest_pair = [[ut_list[ind] for ind in ind_list] for ind_list, ut_list in zip(closest_pair_id, shuffled_list)]                             
        llm_closest_pair_avg_ebd = [np.mean(np.array([self.initial_emb[ut] for ut in ut_list]), axis=0) for ut_list in llm_closest_pair]
        
        



        self.ratio = self.get_correct_ratio(llm_closest_pair)
        tmp = self.mode
        self.mode += '_DrP'
        self.embeddings = np.array(llm_closest_pair_avg_ebd)
        print(f'Embedding shape:{self.embeddings.shape}')
        self.evaluate()
        self.mode = tmp
        self.embeddings = None
        self.cloest_ut_list_llm = llm_closest_pair
        self.ratio = 999            
         
    def save_closet_pairs(self):
        # Open the file in write mode
        model_name = self.model_name.split('/')[-1]
        if 'llm' in self.mode: 
            filename = f"../{self.dataset_name}_{self.mode}_{model_name}_{self.seed_value}sd_llm_lsws.csv"
            data = self.cloest_ut_list_llm
        elif 'gd' in self.mode:
            filename = f"../{self.dataset_name}_{self.mode}_{model_name}_{self.seed_value}sd_gd_lsws.csv"
            data = self.cloest_ut_list_gd
        else:
            filename = f"../{self.dataset_name}_{self.mode}_{model_name}_{self.seed_value}sd_lsws.csv"
            data = self.cloest_ut_list
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)           
            # Write each list as a row in the CSV
            writer.writerows(data)
        # print(f"Clostest pairs saved to {filename}")
    
                  
    def get_plain_embeddings(self):
        self.model.eval()
        print(f'The model is: {self.model_name}')
        batch_size = 30
        all_embeddings = []  
        verbose = 0 
         
        
        for i in tqdm(range(0, len(self.ut_list), batch_size)):
            batch_ut = self.ut_list[i:i + batch_size]
            if 'instructor' in self.model_name:
                if 'bank' in self.dataset_name:
                    sentences = [['Represent the bank purpose for retrieval: ',ut] for ut in batch_ut]
                elif 'clinc150' in self.dataset_name or 'sgd' in self.dataset_name or 'massive' in self.dataset_name:
                    sentences = [['Represent the sentence for retrieving the purpose: ',ut] for ut in batch_ut]
                elif 'mtop' in self.dataset_name:
                    sentences = [['Represent the sentence for retrieval: ',ut] for ut in batch_ut]

                utterance_embeddings = self.model.encode(sentences)
                utterance_embeddings = torch.from_numpy(utterance_embeddings)
            else:
                begin_words =  self.tokenizer.bos_token or "" 
                end_words = self.tokenizer.eos_token or "" 
                if 'sentence-transformers' not in self.model_name and 'e5-large' not in model_name:
                    inputs = self.tokenizer([begin_words + i + end_words for i in batch_ut], padding=True, return_tensors="pt")
                else:
                    if 'e5-large' in model_name:
                        batch_ut = ['query: ' + ut for ut in batch_ut]
                        if verbose == 0:
                            print(f'batch_ut: {batch_ut[0]}')
                            verbose = 2
                    
                    inputs = self.tokenizer(batch_ut, padding=True, return_tensors="pt")

                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

                attention_mask = inputs['attention_mask']
                with torch.no_grad():
                    if self.pooling == 'last':
                        utterance_embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                    elif self.pooling == 'mean':
                        utterance_embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]   
                        pooled = torch.sum(utterance_embeddings * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
                        pooled.masked_fill_(torch.isnan(pooled), 0)    
                        utterance_embeddings = pooled
                utterance_embeddings = utterance_embeddings.cpu()
            all_embeddings.append(utterance_embeddings)  # Append the embeddings of the current batch
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings.float()  # or .double() for float64
        self.embeddings =  all_embeddings.numpy()        
            
            
    def get_sum_embeddings(self):
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.pooling = 'last'
        template = 'The task is intent detection. The goal is to identify the purpose or goal behind a user input. The user intent of this_sentence_:_"*sent_0*"_means_in_one_word:"'

        print(f'The model is: {self.model_name}')
        batch_size = 30
        all_embeddings = []  # Initialize an empty list to store embeddings from each batch
        verbose = 0 
        begin_words = self.tokenizer.bos_token or "" 

        for i in tqdm(range(0, len(self.ut_list), batch_size)):

            batch_ut = self.ut_list[i:i + batch_size]
                
            inputs = self.tokenizer([begin_words + template.replace('*sent_0*', i).replace('_', ' ') for i in batch_ut], padding=True, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            with torch.no_grad():
                utterance_embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                utterance_embeddings.masked_fill_(torch.isnan(utterance_embeddings), 0)   
            utterance_embeddings = utterance_embeddings.cpu()
            all_embeddings.append(utterance_embeddings)  # Append the embeddings of the current batch


        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings.float()  # or .double() for float64
        self.embeddings =  all_embeddings.numpy()
    def get_echo_embeddings(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None      
        if 'gemma' in self.model_name:
            batch_size = 10
        else:
            batch_size = 20

        begin_words = self.tokenizer.bos_token or "" 
        end_words = self.tokenizer.eos_token or ""      
        whole_sent = begin_words +'Instruct:{!%%prompt%%,}\nUser utterance:{!%%text%%}\nUser utterance again:{%%text%%}' + "{" + end_words + "}"
        self.templates = {
            'utterance': whole_sent,
        }           
        model = EchoEmbeddingsMistral.from_pretrained(self.model_name,torch_dtype=torch.bfloat16,
                                                      token= 'hf_ID',
                                                      device_map='auto')
        
        prompt = 'The task is intent detection. The goal is to identify the purpose or goal behind a user input. Give the user intent of the utterance'

        # Create the parser 
        parser = EchoParser(self.model_name, self.templates, max_length=300)
        # Create the pooling: strategy can either be mean or last
        pooling = EchoPooling(strategy=self.pooling)
        all_embeddings = []  # Initialize an empty list to store embeddings from each batch

        verbose = 0
        for i in tqdm(range(0, len(self.ut_list), batch_size)):    
            batch_ut = self.ut_list[i:i + batch_size]
            utterance_variables = [{'prompt': prompt,'text': ut} for ut in batch_ut]
            utterance_tagged = [('utterance', ut) for ut in utterance_variables]

            with torch.no_grad():
                parser_output = parser(utterance_tagged)
                model_output = model(parser_output)
                utterance_embeddings = pooling(model_output)['sentence_embedding']
                utterance_embeddings = utterance_embeddings.cpu()
            all_embeddings.append(utterance_embeddings)  # Append the embeddings of the current batch
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings.float()  # or .double() for float64
        self.embeddings =  all_embeddings.numpy()
        del model 
        torch.cuda.empty_cache()
    
    @staticmethod
    def hungray_aligment(y_true, y_pred):
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D))
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
        return ind, w
    @staticmethod
    def clustering_accuracy(true_labels, cluster_labels):
        contingency_matrix = cluster.contingency_matrix(true_labels, cluster_labels)
        row_ind, col_ind = linear_sum_assignment(contingency_matrix, maximize=True)
        optimal_assignment = contingency_matrix[row_ind, col_ind].sum()
        accuracy = optimal_assignment / len(true_labels)


        ind, w = GetEmbeddings.hungray_aligment(true_labels, cluster_labels)
        acc = sum([w[i, j] for i, j in ind]) / cluster_labels.size
        print(f'Our clustering is aligned:{accuracy == acc}')
        return accuracy

    def evaluate(self):
        print(f'**Mode**: {self.mode}')
        if 'clinc150' in self.dataset_name:
            n_clusters = 150
        elif 'bank77' in self.dataset_name:
            n_clusters = 77
        elif 'sgd' in self.dataset_name:
            n_clusters = 46
        elif 'mtop' in self.dataset_name:
            n_clusters = 102
        elif 'massive' in self.dataset_name:
            n_clusters = 59        
        kmeans = KMeans(n_clusters=n_clusters, random_state = self.seed_value)
        clusters = kmeans.fit_predict(self.embeddings)


        label_encoder = LabelEncoder()
        label_ids = label_encoder.fit_transform(self.intent_list)
        # Evaluate the clustering
        nmi = normalized_mutual_info_score(label_ids, clusters)
        acc = GetEmbeddings.clustering_accuracy(label_ids, clusters)
        

        print(f"Ratio: {self.ratio * 100:.2f}%")
        print(f'NMI: {nmi:.2f}')
        print(f"Clustering Accuracy: {acc * 100:.2f}%")
        print(f'self.embeddings shape: {self.embeddings.shape}')
        # print(f'The length of real label is {len(label_ids)}; the length of input is {len(clusters)}')

        self.save_metrics_to_csv(nmi, acc)

        
    def save_metrics_to_csv(self, nmi, acc):
        # Dictionary of metric results
        model_name = self.model_name.split('/')[-1]    
        file_name=f'../{self.dataset_name}_{model_name}_{self.seed_value}sd.csv'

        metrics = {
            'Model': [model_name],
            'Mode':[self.mode],
            'n':[self.n],
            'ltop':[self.top_conf],
            'Seed': [self.seed_value],
            'Gd_num_1stage':[self.gd_num],
            'LLM_mistake':[self.mistake_count],            
            'Correct_Ratio': [self.ratio],
            'NMI': [nmi],
            'LLM_gen_num':[self.llm_gen_num],
            'Clustering Accuracy': [acc * 100],
        }

        # Write metrics to CSV
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow({key: value[0] for key, value in metrics.items()})




parser = argparse.ArgumentParser(description='Prompting LLM for improvement of embeddings')
parser.add_argument('--seed_value', type=int,default = 1)
parser.add_argument('--dataset_name', type=str, default = 'bank77')
parser.add_argument('--model_name', type=str, default = 'hkunlp/instructor-large')
parser.add_argument('--metric', type=str, default = 'cosine')
parser.add_argument('--first_selection_num', type=int,default = 18)
parser.add_argument('--top_conf', type=int,default = 1)

parser.add_argument('--sampling', type=str, default = 'yes')
parser.add_argument('--llm_for_encoder', type=str,default = 'google/gemma-2-9b-it')


args = parser.parse_args()
seed_value = args.seed_value
model_name = args.model_name
dataset_name = args.dataset_name
metric = args.metric
first_selection_num = args.first_selection_num
top_conf = args.top_conf
sampling = args.sampling
llm_for_encoder = args.llm_for_encoder


# Reproducible

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False    

def run_experiment(seed_value = seed_value, mode='Plain'):
    def process_embeddings(get_func):
            print(f'**Result** (No neighbor)')
            get_func()          
            get_embd.evaluate()   
            print(f'**Result** (Neighbor)')
            get_embd.find_closest_sentences(input_collection = get_embd.ut_list)
            get_embd.save_closet_pairs()

            print(f'**Result** (LLMneighbor)')
            get_embd.find_closest_sentences_llm()

            get_embd.save_closet_pairs()
            print(f'**Result** (GDneighbor)')
            get_embd.find_closest_sentences_gd(input_collection = get_embd.ut_list)


     
    
    get_embd = GetEmbeddings(dataset_name = dataset_name,
                            seed_value=seed_value, 
                            mode = mode,
                            metric = metric,
                            model_name=model_name,
                            first_selection_num = first_selection_num,
                            top_conf = top_conf,
                            sampling = sampling,
                            llm_for_encoder = llm_for_encoder)
    get_embd.load_data()
        
    if mode == 'Sum':
        print(f'**Result** (mode: {mode})')
        process_embeddings(get_embd.get_sum_embeddings)
    elif mode == 'Echo':
        print(f'**Result** (mode: {mode})')        
        process_embeddings(get_embd.get_echo_embeddings)
    elif mode == 'Baseline':
        print(f'**Result** (mode: {mode}-Encoder)')       
        process_embeddings(get_embd.get_plain_embeddings)
    
start_time = time.time()

set_seed(seed_value)
if 'sentence-transformers' in model_name or 'instructor' in model_name or 'e5-large' in model_name:
    print(f'llm_for_ecoder: {llm_for_encoder}')

    modes = ['Baseline']   
else:
    modes = ['Sum','Echo']


for mode in modes:
    print(f'**first_selection_num**: {first_selection_num}; **top_conf**: {top_conf}')
    run_experiment(seed_value, mode = mode)    
    



 
# Record end time
end_time = time.time()



# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
