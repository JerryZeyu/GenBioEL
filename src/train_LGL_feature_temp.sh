

DEVICE_NUMBER=1
MODEL_NAME=LGL_withPrompt_Feature
DATASET=benchmarks

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/lgl_withPrompt_feature \
                                            -model_load_path facebook/bart-large \
                                            -model_token_path facebook/bart-large \
                                            -model_save_path ./model_checkpoints/$MODEL_NAME \
                                            -save_steps 160000 \
                                            -logging_path ./logs/$MODEL_NAME \
                                            -logging_steps 100 \
                                            -init_lr 1e-05 \
                                            -per_device_train_batch_size 8 \
                                            -evaluation_strategy no \
                                            -label_smoothing_factor 0.1 \
                                            -max_grad_norm 0.1 \
                                            -max_steps 160000 \
					                        -warmup_steps 1000 \
                                            -weight_decay 0.01 \
					                        -rdrop 0.0 \
                                            -lr_scheduler_type polynomial \
                                            -attention_dropout 0.1  \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -max_position_embeddings 1024 \
                                            -seed 0 \
					                        -finetune \
					                        -prefix_mention_is
