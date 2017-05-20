home_dir=/home/hsienchin
node2vec=${home_dir}/node2vec/src/main.py
feature_dir=${home_dir}/transfer_learning_tag_detection/feature
echo $node2vec
echo $feature_dir
for feature in `ls ${feature_dir}/*_edge_list`; do
    output_file=${feature_dir}/`basename ${feature}`.emd
    cmd="python ${node2vec} --input ${feature} --output ${output_file} --weighted"
    echo $cmd
    $cmd
done
