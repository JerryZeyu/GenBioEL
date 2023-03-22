

DEVICE_NUMBER=1
MODEL_NAME=GWN_withPrompt_Refined_Country
DATASET=benchmarks

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/gwn_withPrompt_Refined_country \
                                            -model_token_path facebook/bart-large \
                                            -evaluation \
					                        -dict_path $DATASET/gwn_withPrompt_Refined_country/target_kb.json \
                                            -trie_path $DATASET/gwn_withPrompt_Refined_country/trie.pkl  \
                                            -per_device_eval_batch_size 1 \
					                        -model_load_path ./model_checkpoints/$MODEL_NAME/checkpoint-40000 \
                                            -max_position_embeddings 1024 \
					                        -seed 0 \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -prefix_prompt \
                                            -num_beams 5 \
                                            -max_length 2014 \
                                            -min_length 1 \
                                            -dropout 0.1 \
                                            -attention_dropout 0.1 \
                                            -prefix_mention_is \
                                            -testset