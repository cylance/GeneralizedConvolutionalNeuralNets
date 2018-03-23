NUM_NEIGHBORS=10
python sample_modelnet10.py dataSets/ModelNet10csv1000/ 1000
python statistics.py dataSets/ModelNet10csv1000/ $NUM_NEIGHBORS

python csv_to_tf_records.py dataSets/ModelNet10csv1000/ "train"
python csv_to_tf_records.py dataSets/ModelNet10csv1000/ "test"
