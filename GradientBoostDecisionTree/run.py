# -*- coding:utf-8 -*-

from data import DataSet
from model import GBDT

if __name__ == '__main__':
    data_file = 'credit.data.csv'
    dateset = DataSet(data_file)
    gbdt = GBDT(max_iter=20, sample_rate=0.8, learn_rate=0.5, max_depth=7, loss_type='binary-classification')
    gbdt.fit(dateset, dateset.get_instances_idset())
'''
iter1 : train loss=0.387395
iter2 : train loss=0.237866
iter3 : train loss=0.171117
iter4 : train loss=0.117086
iter5 : train loss=0.086727
iter6 : train loss=0.070473
iter7 : train loss=0.057424
iter8 : train loss=0.044750
iter9 : train loss=0.037400
iter10 : train loss=0.028698
iter11 : train loss=0.022619
iter12 : train loss=0.020251
iter13 : train loss=0.017965
iter14 : train loss=0.014910
iter15 : train loss=0.011740
iter16 : train loss=0.010367
iter17 : train loss=0.008023
iter18 : train loss=0.007385
iter19 : train loss=0.005959
iter20 : train loss=0.004837
'''