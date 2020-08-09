# grayscale
python ./train.py --use_gpu --model watson --net dct --colorspace Gray --name gray_watson_dct_trial0 --trainloss weightedsigmoid --nepoch 10 --nepoch_decay 10 --lr 0.0002
python ./train.py --use_gpu --model watson --net fft --colorspace Gray --name gray_watson_fft_trial0 --trainloss weightedsigmoid --nepoch 10 --nepoch_decay 10 --lr 0.0002
python ./train.py --use_gpu --model net-lin --net vgg  --colorspace Gray --name gray_pnet_lin_vgg_trial0 --trainloss ranked
python ./train.py --use_gpu --model net-lin --net squeeze  --colorspace Gray --name gray_pnet_lin_squeeze_trial0 --trainloss ranked 

# color
python ./train.py --use_gpu --model watson --net dct --colorspace RGB --name rgb_watson_dct_trial0 --trainloss weightedsigmoid --nepoch 10 --nepoch_decay 10 --lr 0.0002
python ./train.py --use_gpu --model watson --net fft --colorspace RGB --name rgb_watson_fft_trial0 --trainloss weightedsigmoid --nepoch 10 --nepoch_decay 10 --lr 0.0002
python ./train.py --use_gpu --model net-lin --net vgg  --colorspace RGB --name rgb_pnet_lin_vgg_trial0 --trainloss ranked
python ./train.py --use_gpu --model net-lin --net squeeze  --colorspace RGB --name rgb_pnet_lin_squeeze_trial0 --trainloss ranked 


python ./train.py --use_gpu --model adaptive --colorspace RGB --name rgb_adaptive_trial0 --trainloss weightedsigmoid --nepoch 10 --nepoch_decay 10 --lr 0.0002
python ./train.py --use_gpu --model adaptive --colorspace Gray --name gray_adaptive_trial0 --trainloss weightedsigmoid --nepoch 10 --nepoch_decay 10 --lr 0.0002
