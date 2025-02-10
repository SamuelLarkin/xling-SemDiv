#!/usr/bin/env bash

#############################################################################
#                                                                           #
#                                                                           #
#        Detecting Fine-Grained Cross-Lingual Semantic Divergences          #
#               without Supervision by Learning To Rank                     #
#                                                                           #
#                              eleftheria                                   #
#                                                                           #
#                          ====  Step 4  ====                               #
#                                                                           #
#                         Evaluation on REFreSD                             #
#                                                                           #
#                                                                           #
#############################################################################

##############################################################################


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/clip-scratch/ebriakou/anaconda3/lib

readonly corpus=WikiMatrix                                   #   Corpus from which seed equivalents are extracted
readonly sampling_method=contrastive_divergence_ranking      #   Sampling method for extracting divergent examples from seeds
readonly size=50000                                          #   Number of seeds sampled from original corpus
readonly src=en                                              #   Source language (language code)
readonly tgt=fr                                              #   Target language (language code)
readonly divergent_list=rdpg                                 #   List of divergences (e.g, 'rd' if divergences include
                                                    #                   phrase replacement and subtree deletion)

#############################################################################

readonly root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly data_dir=$root_dir/data
readonly scripts_dir=$root_dir/source

readonly exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
readonly data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
readonly output_dir=$root_dir/trained_bert/$exp_identifier
readonly REFreSD_dir=$root_dir/REFreSD_no_normal/REFreSD_for_huggingface
readonly model=bert-base-multilingual-cased

################################################################################
#                         Synthetic test evaluation                            #
################################################################################

test_set=test_synthetic

time python "$scripts_dir/run_div_margin.py" \
  --node "$SLURM_NODELIST" \
  --model_type bert_margin \
  --model_name_or_path $model \
  --task_name SemDiv \
  --do_eval \
  --best_checkpoint \
  --evaluation_set $test_set \
  --data_dir "$data_dir/" \
  --output_dir "$output_dir" \
  --synth_data_dir "$data_dir/" \
  --overwrite_cache

#################################################################################
#              REFreSD evaluation --- Divergence VS Equivalence                 #
#################################################################################

test_set="test"

time python "$scripts_dir/run_div_margin.py" \
  --node "$SLURM_NODELIST" \
  --model_type bert_margin \
  --model_name_or_path $model \
  --task_name SemDiv \
  --do_eval \
  --best_checkpoint \
  --evaluation_set $test_set \
  --data_dir "$data_dir/" \
  --output_dir "$output_dir" \
  --synth_data_dir "$REFreSD_dir/" \
  --overwrite_cache

#####################################################################################
#          REFreSD evaluation --- Unrelated VS No meaning difference                #
#####################################################################################

test_set=unrelated

time python "$scripts_dir/run_div_margin.py" \
  --node "$SLURM_NODELIST" \
  --model_type bert_margin \
  --model_name_or_path $model \
  --task_name SemDiv \
  --do_eval \
  --best_checkpoint \
  --evaluation_set $test_set \
  --data_dir "$data_dir/" \
  --output_dir "$output_dir" \
  --synth_data_dir "$REFreSD_dir/" \
  --overwrite_cache

########################################################################################
#          REFreSD evaluation -- Some meaning difference VS No meaning difference      #
########################################################################################

test_set=some_meaning_difference
time python "$scripts_dir/run_div_margin.py" \
  --node "$SLURM_NODELIST" \
  --model_type bert_margin \
  --model_name_or_path $model \
  --task_name SemDiv \
  --do_eval \
  --best_checkpoint \
  --evaluation_set $test_set \
  --data_dir "$data_dir/" \
  --output_dir "$output_dir" \
  --synth_data_dir "$REFreSD_dir/" \
  --overwrite_cache

########################################################################################
#                           Print results                                              #
########################################################################################

echo '> Test synthetic:'
time python "$scripts_dir/sentence_evaluation.py" --dict_dir "$output_dir/" --set_ test_synthetic
echo '> REFreSD (Divergence vs Equivalence):'
time python "$scripts_dir/sentence_evaluation.py" --dict_dir "$output_dir/" --set_ test
echo '> REFreSD (Unrelated vs No meaning difference):'
time python "$scripts_dir/sentence_evaluation.py" --dict_dir "$output_dir/" --set_ unrelated
echo '> REFreSD (Some meaning difference vs No meaning difference):'
time python "$scripts_dir/sentence_evaluation.py" --dict_dir "$output_dir/" --set_ some_meaning_difference
