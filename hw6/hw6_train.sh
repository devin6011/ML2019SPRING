trainX=$1
trainY=$2
testX=$3
dictionary=$4

python train1.py $trainX $trainY $testX $dictionary
python train2.py $trainX $trainY $testX $dictionary
python train3.py $trainX $trainY $testX $dictionary
python train4.py $trainX $trainY $testX $dictionary
python train5.py $trainX $trainY $testX $dictionary
