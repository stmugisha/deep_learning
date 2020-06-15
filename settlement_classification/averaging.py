# Averages predictions from 3 deep learning models

import pandas as pd

sub_mobilenet = pd.read_csv('./submissions/submission.csv')
sub_resnet = pd.read_csv('./submissions/sub_resnet.csv')
sub_cnn = pd.read_csv('./submissions/sub_cnn.csv')
sub_enet = pd.read_csv('./submissions/sub_e-net.csv') #(512px images)

#sub_resnet['Label'] = (sub_resnet['Label'] * 0.75+ sub_mobilenet['Label'] #* 0.2 + sub_rf['Label'] * 0.05)/3

sub_cnn['Label'] = (sub_cnn['Label'] * 0.75+
                    sub_resnet['Label'] * 0.2 +
                    sub_mobilenet['Label']*0.05)/3

print(min(sub_cnn['Label']), max(sub_cnn['Label']))
sub_cnn.to_csv('./submissions/sub_cnnv2.csv', index=False)
