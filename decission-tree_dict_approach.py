from sklearn import datasets

import numpy as np
import math
import pprint
import pandas as pd
import sys
sys.setrecursionlimit(1500)
class decision_tree :
    def entropy(self,col,name):
        tot = col.shape[0]
        uni = col[name].value_counts().rename_axis('unique').reset_index(name='counts')
        uni['p'] = uni['counts']/tot
        # uni['inf'] = uni['p']*(uni['p'].apply(lambda x: x*math.log2(x)))
        uni['p'] = uni['p'].apply(lambda x: -1*(x*math.log2(x)))
        # print(col.shape[0])
        # print(uni)

        return uni['p'].sum()

    def gain(self,x,y,features):
        # x.columns = range(x.shape[1])
        x['out'] = y
        x['ent/out'] = self.entropy(x,'out')
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

        return x,self.find_winner(x)

    def find_winner(self,df):
        features = df.columns.values
        df_details = df[features[5:]]
        df_details_cols = df_details.columns.values

        best_f = {}

        for f in df_details_cols:
            if f.find("gain/") != -1:
                best_f[f.replace("gain/","")] = df[f].unique()[0]

        sorted_dict = list(dict(sorted(best_f.items(),key=lambda item: item[1])).keys())

        return sorted_dict[0]


    def get_subtable(self,df, node, value):
        '''
        Function to get a subtable of met conditions.

        node: Column name
        value: Unique value of the column
        '''
        return df[df[node] == value]

    def dict_print(self,dty):
        print()
        level = dty.get('level')
        print('level :',level)
        count = dty.get('count')
        for k in count.keys():
            print('count of',k,":",count.get(k))
        print("Entropy :",round(dty.get('entropy'),2))
        print('feature :',dty.get('feature'))
        gain = dty.get('gain')
        if gain != None:
            print('gain :',round(dty.get('gain'),2))

    def buildTree(self,df,out,cols,l=0,printer=False,tree={}):

        if len(cols) == 0:
            tree['leaf'] = True
            tree['feature'] = "Reached leaf node"
            tree['level'] = l
            tree['entropy'] = 0.0
            df['out'] = out
            tree['count'] = df['out'].value_counts().to_dict()
            uni = df['out'].unique()

            if len(uni)==1:
                tree['out'] = uni[0]
            else:
                tree['out'] = list(uni)

            if printer:
                self.dict_print(tree)

            return

        df,node = self.gain(df,out,cols)
        # print(node)


        if len(df['out'].unique()) == 1:
            tree['leaf'] = True
            tree['feature'] = "Reached leaf node"
            tree['level'] = l
            tree['entropy'] = 0.0
            tree['out'] = df['out'].unique()[0]
            tree['count'] = df['out'].value_counts().to_dict()

            if printer:
                self.dict_print(tree)

            return


        if tree == {}:
            tree[node] = {'gain':df['gain/'+node].unique()[0],'feature':node,'entropy':df['ent/'+node].unique()[0],'level':l,'count':df['out'].value_counts().to_dict(),'leaf':False}
            tree['leaf'] = False
            if printer:
                self.dict_print(tree[node])

            cols = set(cols).difference(set([node]))
            l += 1
            for f in df[node].unique():
                tree[node][f] = {}
                subtable = self.get_subtable(df, node, f)
                columns = subtable.columns.values

                if printer:
                    self.buildTree(subtable[columns[:4]], subtable[columns[4]], list(cols), l=l, tree=tree[node][f],printer=True)
                else:
                    self.buildTree(subtable[columns[:4]],subtable[columns[4]],list(cols),l=l,tree=tree[node][f])


        return tree

    def fit(self,df,out,printer=False):

        columns = [str(f) for f in df.columns.values]
        self.columns = columns
        df.columns = columns

        if printer:
            self.root = self.buildTree(df, out, columns,printer=True)
        else:
            self.root = self.buildTree(df, out, columns)

        return self.root
    
    def pretty_print(self):
        pprint.pprint(self.root)
        
    def predict(self,dty):

        nroot = self.root.copy()

        cols = [str(i) for i in dty.keys()]
        present = True
        feature = ""
        for c in cols:
            if c not in self.columns:
                feature = c
                present = False
                break

        # pprint.pprint(nroot)


        if present:
            while nroot.get('leaf') == False:
                key = set(list(nroot.keys())).intersection(set(cols))
                key = list(key)[0]
                # print(list(key)[0])
                nroot = nroot.get(key)
                value = dty.get(key)

                if nroot.get(value) != None:
                    nroot = nroot.get(value)
                else:
                    print("the value",value,"is not present in the feature",key ,"for the given combination")
                    return
        else:
            print("the",feature,"is not present in the decission tree")

        return nroot.get('out')










iris_data = datasets.load_iris()
target_names = list(iris_data.target_names)
cols = ["sepal_length", "sepal_width", 'petal_length', 'petal_width']
data = pd.DataFrame(iris_data.data, columns=cols)
# print(data)
target = pd.DataFrame(iris_data.target)[0].apply(lambda x: target_names[x]).to_frame()


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
# print(target)
# print(iris_data.target_names)

data['sl_labeled'] = toLabel(data, 'sepal_length')
data['sw_labeled'] = toLabel(data, 'sepal_width')
data['pl_labeled'] = toLabel(data, 'petal_length')
data['pw_labeled'] = toLabel(data, 'petal_width')

data.drop(cols, axis = 1, inplace = True)
data.columns = cols

columns = data.columns.values
dtree = decision_tree()
# tree = dtree.buildTree(data,target,columns,printer=True)
tree = dtree.fit(data.copy(),target)
# pprint.pprint(tree)
d = {'sepal_length':'c','sepal_width':'c','petal_length':'c','petal_width':'c'}
pred = dtree.predict(d)
print(pred)