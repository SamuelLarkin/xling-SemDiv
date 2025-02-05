#!/usr/bin/env bash

#############################################################################
#                                                                           #
#                                                                           #
#        Detecting Fine-Grained Cross-Lingual Semantic Divergences          #
#               without Supervision by Learning To Rank                     #
#                                                                           #
#                              eleftheria                                   #
#                                                                           #
#                          ====  Step 5  ====                               #
#                                                                           #
#                    Token tag predictions on REFreSD                       #
#                                                                           #
#                                                                           #
#############################################################################

##############################################################################

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/clip-scratch/ebriakou/anaconda3/lib

readonly corpus=WikiMatrix                           #   Corpus from which seed equivalents are extracted
readonly sampling_method=contrastive_multi_hard      #   Sampling method for extracting divergent examples from seeds
readonly size=50000                                  #   Number of seeds sampled from original corpus
readonly src=en                                      #   Source language (language code)
readonly tgt=fr                                      #   Target language (language code)
readonly divergent_list=rdpg                         #   List of divergences (e.g, 'rd' if divergences include
                                            #                  phrase replacement and subtree deletion)

#################################################################################

readonly root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly data_dir=$root_dir/data
readonly scripts_dir=$root_dir/source

readonly exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
readonly data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
readonly output_dir=$root_dir/trained_bert/$exp_identifier
readonly REFreSD_dir=$root_dir/REFreSD/REFreSD_for_huggingface
readonly model=bert-base-multilingual-cased

###################################################################################
#          Token level predictions on REFreSD	                                  #
###################################################################################

readonly test_set="test"

python "$scripts_dir/run_div_multi.py" \
  --node "$SLURM_NODELIST" \
  --model_type SemDivMulti \
  --model_name_or_path $model \
  --task_name SemDiv \
  --do_eval   \
  --best_checkpoint \
  --evaluation_set $test_set \
  --data_dir "$data_dir/"   \
  --output_dir "$output_dir" \
  --synth_data_dir "$REFreSD_dir/" \
  --overwrite_cache