
mkdir dataset

# JND Dataset
#wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/jnd.tar.gz -O ./dataset/jnd.tar.gz --no-check-certificate

#mkdir dataset/jnd
#tar -xzf ./dataset/jnd.tar.gz -C ./dataset
#rm ./dataset/jnd.tar.gz

# 2AFC Val set
mkdir dataset/2afc/
wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/twoafc_val.tar.gz -O ./dataset/twoafc_val.tar.gz wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/jnd.tar.gz -O ./dataset/jnd.tar.gz --no-check-certificate

mkdir dataset/2afc/val
tar -xzf ./dataset/twoafc_val.tar.gz -C ./dataset/2afc
rm ./dataset/twoafc_val.tar.gz

# 2AFC Train set
mkdir dataset/2afc/
wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/twoafc_train.tar.gz -O ./dataset/twoafc_train.tar.gz wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/jnd.tar.gz -O ./dataset/jnd.tar.gz --no-check-certificate

mkdir dataset/2afc/train
tar -xzf ./dataset/twoafc_train.tar.gz -C ./dataset/2afc
rm ./dataset/twoafc_train.tar.gz
