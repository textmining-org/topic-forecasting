for method_name in dmr lda; do
  for target_name in patent_201701_202312_without_duplicates paper_201701_202312_without_duplicates news_201701_202312_without_duplicates; do
    python3 -u opt_num_topics.py --method_name $method_name --target_name $target_name
  done
done
