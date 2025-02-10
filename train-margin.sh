#!/usr/bin/env bash


#############################################################################
#                                                                           #
#                                                                           #
#        Detecting Fine-Grained Cross-Lingual Semantic Divergences          #
#               without Supervision by Learning To Rank                     #
#                                                                           #
#                              eleftheria                                   #
#                                                                           #
#                          ====  Step 3  ====                               #
#                                                                           #
#                 Fine-tune divergnetmBERT using divergence ranking         #
#                                                                           #
#                                                                           #
#############################################################################

##############################################################################
 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/clip-scratch/ebriakou/anaconda3/lib

readonly corpus=WikiMatrix                                   #   Corpus from which seed equivalents are extracted
readonly sampling_method=contrastive_divergence_ranking      #   Sampling method for extracting divergent examples from seeds
readonly size=50000                                          #   Number of seeds sampled from original corpus
readonly src=$1                                              #   Source language (language code)
readonly tgt=$2                                              #   Target language (language code)
readonly divergent_list=rdpg                                 #   List of divergences (e.g, 'rd' if divergences include
                                                    #                      phrase replacement and subtree deletion)
readonly lr=2e-5                                             #   Learning rate
readonly batch_size=16                                       #   Training batch size
readonly epochs=5                                            #   Number of training epochs
readonly margin=5                                            #   Margin used in the training loss

#############################################################################

readonly root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly data_dir=$root_dir/data
readonly scripts_dir=$root_dir/source

readonly exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
readonly data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
readonly output_dir=$root_dir/trained_bert_new/$exp_identifier
readonly model=bert-base-multilingual-cased

################################################################################

# Create output directory if not exist
if [[ ! -e $output_dir ]]; then
  mkdir -p "$output_dir"
elif [[ ! -d $output_dir ]]; then
  echo "$output_dir already exists but is not a directory" 1>&2
fi


time python "${scripts_dir}/run_div_margin.py" \
  --data_dir "${data_dir}/" \
  --do_eval \
  --do_train \
  --evaluate_during_training \
  --evaluate_on_training \
  --learning_rate ${lr} \
  --logging_steps 100 \
  --margin "${margin}" \
  --max_seq_length 128 \
  --model_name_or_path "${model}" \
  --model_type bert_margin \
  --node "$SLURM_NODELIST" \
  --num_train_epochs ${epochs} \
  --output_dir "${output_dir}" \
  --overwrite_cache \
  --overwrite_output_dir \
  --per_gpu_train_batch_size=${batch_size} \
  --save_steps 100 \
  --synth_data_dir "${data_dir}/" \
  --task_name SemDiv
