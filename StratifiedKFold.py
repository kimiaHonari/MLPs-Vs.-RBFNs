
import random

import numpy as np
class StratifiedKFold:
    def __init__(self, n_splits=10):
        if not isinstance(n_splits, int):
            raise ValueError('The number of folds must be Integer. ')
        if n_splits <= 1:
            raise ValueError(
                "The number of fold must be equal or more than 2.")


        self.n_splits=int(n_splits)



    def generate_fold(self,X,Y):

        unique_elements, counts_elements = np.unique(Y, return_counts=True)

        y_counts = counts_elements
        min_groups = np.min(y_counts)
        print(y_counts)
        if self.n_splits > min_groups:
            raise ValueError('The number of folds must be more than the number of members in each class ')
        print(unique_elements)
        self.x_train=X
        print(Y.shape)
        self.y_train=np.reshape(Y,50000)
        # self.y_train=Y
        self.n_sample=len(Y)/self.n_splits
        self.max_sample_per_class=y_counts/self.n_splits
        indices=np.argsort(self.y_train)
        print(self.y_train)
        print(indices)
        folds=[]
        number=0
        first=1
        remainlist=[]
        start=0
        print(self.max_sample_per_class)
        for classes in unique_elements:
            start=number
            number+=y_counts[classes]
            data=indices[start:number]
            print("data size",len(data))

            random.shuffle(data)
            if first:

                limit = int(self.max_sample_per_class[classes])

                for i in range(0,self.n_splits):
                    folds.append(np.array(data[i*limit:(i+1)*limit]))
                    if (i+1==self.n_splits):
                        remainlist.extend(data[(i+1)*limit:])
                    print("first len fold:",len(folds[i]))
                first=0
            else:
                limit = int(self.max_sample_per_class[classes])
                i=0
                for i in range(0, self.n_splits):
                    if (i + 1 == self.n_splits):
                        remainlist.extend(data[(i + 1) * limit:])
                    folds[i]=np.hstack([folds[i],(np.array(data[i * limit:(i + 1) * limit]))])
                    print("after len fold:", len(folds[i]))


        print("remain size:",len(remainlist))
        random.shuffle(remainlist)

        step=int(len(remainlist)/self.n_splits)
        print(step)
        if step==0:
            self.folds = folds
            return folds

        for i in range(0, self.n_splits):
            if (i + 1 == self.n_splits):

                folds[i] = np.hstack([folds[i], (np.array(remainlist[i * step:]))])
            else:
                folds[i] = np.hstack([folds[i], (np.array(remainlist[i * step:(i + 1) * step]))])

        self.folds=folds
        return folds

    def pop(self,foldIdx=0):
        if foldIdx>= self.n_splits:
            raise ValueError(
                "The index is out of range")

        mask = np.ones(len(self.x_train), dtype=bool)
        mask[self.folds[foldIdx],] = False
        x_validation, x_train = self.x_train[~mask], self.x_train[mask]

        mask = np.ones(len(self.y_train), dtype=bool)
        mask[self.folds[foldIdx],] = False
        y_validation, y_train = self.y_train[~mask], self.y_train[mask]

        return (x_train,y_train),(x_validation,y_validation)

