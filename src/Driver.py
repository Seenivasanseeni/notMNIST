import Dataset
import Model

def train(dat,Mod):
    iterations=35

    for i in range(iterations):
        images,labels=dat.getBatch()
        lo,acc=Mod.train(images,labels)
        print("Iteration:",i,"Loss",lo,"Acc",acc)
        import pdb
        pdb.set_trace()
    return

def test(dat,Mod):
    pass


def main():
    dat=Dataset.Dataset()
    Mod=Model.Model()
    Mod.initializeModel()
    train(dat,Mod)
    return

if __name__ == '__main__':
    main()
