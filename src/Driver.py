import Dataset
import Model

def train(dat,Mod):
    iterations=1500
    printStep=20
    for i in range(iterations):
        images,labels=dat.getBatch()
        lo,acc=Mod.train(images,labels)
        if(i%printStep==0):
           # print("Iteration:",i,"Loss",lo,"Acc",acc)
            test(dat,Mod)
    return

def test(dat,Mod):
    images,labels=dat.getBatch(True)
    lo,acc=Mod.test(images,labels)
    print("TEST","LOSS",lo,"ACCURACY",acc)
    return

def main():
    dat=Dataset.Dataset()
    Mod=Model.Model()
    Mod.initializeModel()
    train(dat,Mod)
    return

if __name__ == '__main__':
    main()
