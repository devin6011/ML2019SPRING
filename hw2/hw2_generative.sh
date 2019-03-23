rawData=$1
testData=$2
providedTrainFeature=$3
providedTrainLabel=$4
providedTestFeature=$5
output=$6

python generativePredict.py $providedTestFeature $output
