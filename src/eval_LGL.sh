

DEVICE_NUMBER=$1
MODEL_NAME=$2
DATASET=$3

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/lgl_withOnlyMention \
                                            -model_token_path facebook/bart-large \
                                            -evaluation \
					                        -dict_path $DATASET/lgl_withOnlyMention/target_kb.json \
                                            -trie_path $DATASET/lgl_withOnlyMention/trie.pkl  \
                                            -per_device_eval_batch_size 1 \
					                        -model_load_path ./model_checkpoints/$MODEL_NAME/checkpoint-20000 \
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