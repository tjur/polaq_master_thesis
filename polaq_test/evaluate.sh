export PREDICTIONS=$1
python3 evaluate-v1.1.py data/polaq_dataset_test.json $PREDICTIONS
