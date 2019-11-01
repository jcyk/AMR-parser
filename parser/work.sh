dataset=../preprocessing/2017
python3 work.py --test_data ${dataset}/dev.txt_processed_preprocess \
               --test_batch_size 4444 \
               --load_path ckpt \
               --sense_table ${dataset}/sense_table \
               --wiki_table ${dataset}/wiki_table\
               --beam_size 8\
               --alpha 0.6\
               --max_time_step 100\
               --output_suffix _test_out