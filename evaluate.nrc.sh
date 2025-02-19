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

readonly model=bert-base-multilingual-cased
readonly corpus=WikiMatrix                                   #   Corpus from which seed equivalents are extracted
readonly size=50000                                          #   Number of seeds sampled from original corpus
readonly src=en                                              #   Source language (language code)
readonly tgt=fr                                              #   Target language (language code)
readonly divergent_list=rdpg                                 #   List of divergences (e.g, 'rd' if divergences include
                                                             #     phrase replacement and subtree deletion)

#############################################################################

readonly root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly scripts_dir=$root_dir/source
# readonly REFreSD_dir=$root_dir/REFreSD/REFreSD_for_huggingface
readonly REFreSD_dir=$root_dir/REFreSD/NRC

function sentence_level {
  local -r sampling_method=$1
  local -r test_set="test"

  local -r exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
  local -r data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
  # local -r model_dir=$root_dir/trained_bert_new/$exp_identifier${SLURM_JOBID:+.$SLURM_JOBID}
  local -r model_dir=$root_dir/trained_bert/${exp_identifier}

  command time --portability python "$scripts_dir/run_div_margin.py" \
    --best_checkpoint \
    --data_dir "$data_dir/" \
    --do_eval \
    --evaluation_set $test_set \
    --model_name_or_path $model \
    --model_type bert_margin \
    --node "$SLURM_NODELIST" \
    --output_dir "$model_dir" \
    --overwrite_cache \
    --synth_data_dir "$REFreSD_dir/" \
    --task_name SemDiv

  ########################################################################################
  #                           Print results                                              #
  ########################################################################################

  echo '> REFreSD (Divergence vs Equivalence):'
  command time --portability python "$scripts_dir/sentence_evaluation.py" --dict_dir "$output_dir/" --set_ $test_set
}


function token_level {
  local -r sampling_method=$1
  local -r test_set="test"

  local -r exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
  local -r data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
  # local -r model_dir=$root_dir/trained_bert/$exp_identifier${SLURM_JOBID:+.$SLURM_JOBID}
  local -r model_dir=$root_dir/trained_bert/${exp_identifier}

  command time --portability python "$scripts_dir/run_div_multi.py" \
    --best_checkpoint \
    --data_dir "$data_dir/" \
    --do_eval \
    --evaluation_set $test_set \
    --model_name_or_path $model \
    --model_type SemDivMulti \
    --node "$SLURM_NODELIST" \
    --output_dir "$model_dir" \
    --overwrite_cache \
    --synth_data_dir "$REFreSD_dir/" \
    --task_name SemDiv
}



sentence_level contrastive_divergence_ranking
token_level contrastive_multi_hard
