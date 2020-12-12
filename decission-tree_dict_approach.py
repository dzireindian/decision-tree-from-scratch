from sklearn import datasets

import numpy as np
import math
import pprint
import pandas as pd
import sys
sys.setrecursionlimit(1500)

def entropy(col,name):
    tot = col.shape[0]
    uni = col[name].value_counts().rename_axis('unique').reset_index(name='counts')
    uni['p'] = uni['counts']/tot
    # uni['inf'] = uni['p']*(uni['p'].apply(lambda x: x*math.log2(x)))
    uni['p'] = uni['p'].apply(lambda x: -1*(x*math.log2(x)))
    # print(col.shape[0])
    # print(uni)

    return uni['p'].sum()

def gain(x,y,features):
    # x.columns = range(x.shape[1])
    x['out'] = y
    x['ent/out'] = entropy(x,'out')
    target_values = x['out'].value_counts().to_dict()
    initial_size = x.shape[0]
    for f in features:
        # print(f)
        values = x[f].value_counts().to_dict()
        ent_x = 0
        split_info = 0
        for k1 in values.keys():
            current_size = x[x[f] == k1].shape[0]
            ent1 = 0

            for k2 in target_values.keys():
                df1 = x[(x[f] == k1)&(x['out']==k2)].shape[0]
                # print(k1,k2,df1,current_size,df1/current_size)
                if df1 != 0:
                    ent1 += (-df1/current_size)*math.log2(df1/current_size)

            # print(current_size/initial_size)
            ent_x += (current_size/initial_size)*ent1
            split_info += (-current_size/initial_size)*math.log2(current_size/initial_size)
        x['ent/'+f] = ent_x
        x["inf/"+f] = x['ent/out']-ent_x
        x['gain/'+f] = x['inf/'+f]/split_info
        # print(x['gain/'+f])

    return x,find_winner(x)

def find_winner(df):
    features = df.columns.values
    df_details = df[features[5:]]
    df_details_cols = df_details.columns.values

    best_f = {}

    for f in df_details_cols:
        if f.find("gain/") != -1:
            best_f[f.replace("gain/","")] = df[f].unique()[0]

    sorted_dict = list(dict(sorted(best_f.items(),key=lambda item: item[1])).keys())

    return sorted_dict[0]


def get_subtable(df, node, value):
    '''
    Function to get a subtable of met conditions.

    node: Column name
    value: Unique value of the column
    '''
    return df[df[node] == value]

def buildTree(df,out,cols,tree={}):

    df,node = gain(df,out,cols)
    # print(node)

    attValue = np.unique(df[node])

    if tree == {}:
        tree[node] = {'gain':df['gain/'+node].unique()[0],'feature':node,'entropy':df['gain/'+node].unique()[0],'count':df['out'].value_counts().to_dict()}
        for f in df[node].unique():
            cols = set(cols).difference(set([node]))
            if len(cols) == 0:
                tree['leaf'] = True
                return
            elif len(df['out'].unique()) == 1:
                tree['leaf'] = True
                return
            else:
                tree[node][f] = {}
                subtable = get_subtable(df, node, f)
                columns = subtable.columns.values
                buildTree(subtable[columns[:4]],subtable[columns[4]],list(cols),tree[node][f])

    # for value in attValue:
    #
    #     subtable = get_subtable(df, node, value)
    #     clValue, counts = np.unique(subtable['out'], return_counts=True)
    #
    #     if len(counts) == 1:
    #         tree[node][value] = clValue[0]
    #     else:
    #         features = subtable.columns.values[:5]
    #         # print(features)
    #         table = subtable[features[:5]]
    #         # print(table[features[:4]])
    #         # print(node)
    #         # print(set(features).difference(set([node])))
    #         tree[node][value] = buildTree(table[features[:4]],table[features[4]],[f for f in features if f != node],tree)  # Calling the function recursively

    return tree

iris_data = datasets.load_iris()
data = pd.DataFrame(iris_data.data, columns=["sl", "sw", 'pl', 'pw'])
target = pd.DataFrame(iris_data.target)
# target = target.astype('bool')

def label(val, *boundaries):
    if (val < boundaries[0]):
        return 'a'
    elif (val < boundaries[1]):
        return 'b'
    elif (val < boundaries[2]):
        return 'c'
    else:
        return 'd'

def toLabel(df, old_feature_name):
    second = df[old_feature_name].mean()
    minimum = df[old_feature_name].min()
    first = (minimum + second)/2
    maximum = df[old_feature_name].max()
    third = (maximum + second)/2
    return df[old_feature_name].apply(label, args= (first, second, third))

# print(data)
# print(iris_data.target_names)

data['sl_labeled'] = toLabel(data, 'sl')
data['sw_labeled'] = toLabel(data, 'sw')
data['pl_labeled'] = toLabel(data, 'pl')
data['pw_labeled'] = toLabel(data, 'pw')

data.drop(['sl', 'sw', 'pl', 'pw'], axis = 1, inplace = True)

columns = data.columns.values
target = toLabel(target,0)

tree = buildTree(data,target,columns)
pprint.pprint(tree)