from Dataset import Dataset,readFromPath

m=Dataset()
trD,teD=m.trainData,m.testData
images,labels=m.getBatch()
import pdb
pdb.set_trace()
im,la=m.getBatch()
