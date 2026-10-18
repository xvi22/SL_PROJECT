# SL_PROJECT layout

Run every script FROM THIS MAIN FOLDER, for example:
    python scripts\week2_extraction\clean_hands.py

scripts\
  week1_word_selection\   find_word, count_words, pick_words, save_word_list
  week2_extraction\       test_one_clip, extract_all (v1), check_clips (v1),
                           extract_all_v2, check_clips_v2, compare_v1_v2,
                           find_phantoms, clean_hands (v3)
  week3_training\         preprocess, split, train_baseline
  week5_live\             mirror_frame (left-handed mode)

data\
  week1\    word_counts.csv, word_list.csv (the 80 words)
  test\     landmarks_test\ (one-clip test)
  v1\       landmarks\, landmarks_index.csv, span_stats.csv, checks\   (record only)
  v2\       landmarks_v2\, index, span stats, checks_v2\, compare_v1_v2.csv,
            phantom_stats.csv, phantoms\                               (record only)
  v3\       landmarks_v3\, landmarks_index_v3.csv, clean_checks\      (FINAL dataset)
  processed_v1\, split_v1.csv, video_sizes.csv                         (Week 3)

report_outputs\   script outputs for the report appendix
runs\             training runs (Week 3 onwards)
models\           MediaPipe model files
wlasl-processed\, WLASL-master\   original WLASL data (untouched)
backup_before_tidy.zip   copy of all scripts and CSVs from before the tidy-up
