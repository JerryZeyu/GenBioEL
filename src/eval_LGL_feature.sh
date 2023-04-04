

DEVICE_NUMBER=0
MODEL_NAME=LGL_withPrompt_Feature_temp
DATASET=benchmarks

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/lgl_withPrompt_feature/test \
                                            -model_token_path facebook/bart-large \
                                            -evaluation \
					                        -dict_path $DATASET/lgl_withPrompt_feature/target_kb.json \
                                            -trie_path $DATASET/lgl_withPrompt_feature/trie.pkl  \
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