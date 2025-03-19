

# SPILL: Zero-shot Intent Clustering with Large Language Models

## Introduction

### What is SPILL?
It is a none-fined and zero-shot appraoch to further boosting the peroformance of existing embedders for clustering tasks. The main idea is to see a clustering task as a small-scale selection problem. Based on this perspective, we can use LLM to make good selection. Good selection leads to good clustering performance

### In which scenario you can use SPILL?
If you have an unlabeled dataset and want to group it based on shared similarities (such as intent, attitude, or other characteristics), **SPILL** can be used to group these datasets more effectively by refining the embeddings of each example in the dataset. Note: Since our study focuses on intent clustering, if you wish to use other clustering types, you will need to modify the prompt accordingly.

### How to decide cluster number?
In our study, we assume a known number of clusters to compare with other baselines. However, in reality, the number of clusters is usually unknown. You can either define the number of clusters yourself or use the silhouette score to determine the optimal number of clusters.

### After I get predicted clusters, what will be next?
You can try randomly select some examples with in each cluster and ask LLM to summrize the pattern for you.


## Scripts to run SPILL

The script run_spill.py is used for our experiments. Note that echo_embeddings.py is one of the baselines embedder we compare, called **Echo**. The script for Echo is sourced from the following repository:
https://github.com/jakespringer/echo-embeddings
Note that if you want to custimize to your work (e,g, change prompt for your desired task or get cluster labels), you have to modify the script yourself.

### Before runing run_spill.py: Steps to Follow
1. **Download Required Scripts:** run_spill.py, echo_embeddings.py, and requirements.txt
2. **Install the package from the requirements.txt**
3. **Set Filepaths in the Script:** In the script you have to put the datasets filepath into filepath you set, and set a filepath to save output csvfile. Replace ".." with your filepath in the code.
You can download from below. 
    
    clinc150, mtop and massive: We use the same dataset settings as one of baselines we compare, **ClusterLLM**: https://github.com/zhang-yu-wei/ClusterLLM

    bank77: Huggingface (This one is in the script and will be automatically download if you choose experiement on the dataset)
    
4. **Add huggingface token ID:** for using some models like LLama or Gemma, replace ''hf_ID' with yours in the code.


Then you are good to go!



### The arguments are used in our experiements
1. seed_value: In our study we use seed from 1 to 5.
2. dataset_name: bank77, clinc150, massive, mtop 
3. model_name: the model we experiemtn with, the full huggingface model id is required
4. sampling: this is legacy argument, use **yes** will be fine.
5. first_selection_num: it means **tolerance** in our paper for the first stage. (we use 20 in our study)
6. top_conf: i.e. $l_{top}$ in our paper (we use 14 in our study)
7. metric: we use euclidean
8. llm_for_encoder: if an **encdoer** is used for the model_name, we need an LLM for the 2nd stage selection. **Decoder** will use itselve to do selection


An example running:

For an **decoder** embedder ( the 2nd stage will also use it)

    python run_spill.py \
      --seed_value 1 \
      --dataset_name bank77 \
      --model_name meta-llama/Llama-3.1-8B-Instruct \
      --sampling yes \
      --first_selection_num 20 \
      --top_conf 14 \
      --metric euclidean \
      
For an **encoder** embedder ( the 2nd stage will need an LLM, the example below use LLama)

    python run_spill.py \
      --seed_value 1 \
      --dataset_name bank77 \
      --model_name hkunlp/instructor-large \
      --sampling yes \
      --first_selection_num 20 \
      --top_conf 14 \
      --metric euclidean \
      --llm_for_encoder meta-llama/Llama-3.1-8B-Instruct\

### After running the script: 
You will get two csv files, one is the utterances selected by different stage. The other is more important. It give you metric results: the NMI and Clustering accuracy (Also other column but not for the main results). Check the column Mode in the csv file. The Mode will have the follwoing words:

If the embeder is **encoder**, you will see the following names in the file:
1. if it is called Baseline, it means the seed embedding is directly use for clustering. (We present in our main result as plain)
2. if it includes words _RdDrP, it means the seed is pooling with all of the first stage selection (We present in our ablation about contribution of two stages)
3. if it includes words _OdrDrP, it means the seed is pooling with top 10 closest utterances (This is just for reference, not presented in our paper)
4. if it includes words _llmCoT, it means the seed is pooling with the utterances after 2nd stage selection. (We present in our main result)
5. if it includes words_gdCoT, it means the seed is pooling with the 100% correct selection ratio from the first stage selection (We use 20 and we present in our ablation).

If the embedder is **decoder**, you will see:
1. if it is called Sum or Echo, it means the seed embedding is directly use for clustering derived from either Sum or Echo prompt. (We present in our main result as Summarizer or Echo)
Other are the same as encoder item 2, 3, 4, 5. only with a prefix either Echo or Sum



