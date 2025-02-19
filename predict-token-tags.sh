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


readonly corpus=WikiMatrix                           #   Corpus from which seed equivalents are extracted
readonly sampling_method=contrastive_multi_hard      #   Sampling method for extracting divergent examples from seeds
readonly size=50000                                  #   Number of seeds sampled from original corpus
readonly src=en                                      #   Source language (language code)
readonly tgt=fr                                      #   Target language (language code)
readonly divergent_list=rdpg                         #   List of divergences (e.g, 'rd' if divergences include
                                                     #                  phrase replacement and subtree deletion)

#################################################################################

readonly root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# readonly data_dir=$root_dir/data
readonly scripts_dir=$root_dir/source

readonly exp_identifier=from_${corpus}.${src}-${tgt}.tsv.filtered_sample_${size}.moses.seed/${sampling_method}/${divergent_list}
readonly data_dir=$root_dir/for_divergentmBERT/${exp_identifier}
# readonly output_dir=$root_dir/trained_bert/$exp_identifier${SLURM_JOBID:+.$SLURM_JOBID}
readonly output_dir=$root_dir/trained_bert/$exp_identifier
readonly REFreSD_dir=$root_dir/REFreSD/REFreSD_for_huggingface
readonly model=bert-base-multilingual-cased

###################################################################################
#          Token level predictions on REFreSD	                                  #
###################################################################################

# NOTE: Is the prefix of a tsv file under REFreSD/REFreSD_for_huggingface/ aka $REFreSD_dir.
# NOTE: test.tsv is part of the repository.
# NOTE: The tsv file must have:
#   - #label: {0, 1}
#   - #1ID: int
#   - #2ID: int
#   - #english_sentence: str
#   - #french_sentence: str
#   - #english_rational: {1,2,3}+
#   - #french_rationale: {1,2,3}+
readonly test_set="Samuel"

command time --portability python "$scripts_dir/run_div_multi.py" \
  --best_checkpoint \
  --data_dir "$data_dir/" \
  --do_eval \
  --evaluation_set $test_set \
  --model_name_or_path $model \
  --model_type SemDivMulti \
  --node "$SLURM_NODELIST" \
  --output_dir "$output_dir" \
  --overwrite_cache \
  --synth_data_dir "$REFreSD_dir/" \
  --task_name SemDiv
