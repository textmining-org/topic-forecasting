for method_name in dmr; do
  for target_name in patents papers news; do
    python3 -u opt_num_topics.py --method_name $method_name --target_name $target_name
  done
done
