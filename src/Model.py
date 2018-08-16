import tensorflow as tf

class Model():
    def initializeModel(self):
        self.images=tf.placeholder(tf.float32,shape=[None,28,28],name="images")
        self.labels=tf.placeholder(tf.float32,shape=[None,10],name="labels")
        self.keep_prob=tf.placeholder(tf.float32,[1])
        self.images_r=tf.reshape(self.images,shape=[-1,28,28,1],name="ReshapedImage")

        self.conv1=tf.layers.conv2d(self.images_r,32,kernel_size=[5,5],strides=[2,2],padding="VALID",name="conv1")
        self.pool1=tf.layers.max_pooling2d(self.conv1,pool_size=[2,2],strides=[2,2],name="pool1")

        self.conv2=tf.layers.conv2d(self.pool1,64,kernel_size=[5,5])

        self.pool2=tf.layers.max_pooling2d(self.conv2,pool_size=[2,2],strides=[1,1])

        self.flat=tf.layers.flatten(self.pool2,name="flatten")

        self.dropout=self.flat * self.keep_prob

        self.dense1=tf.layers.dense(self.dropout,units=50,name="dense1")

        self.dense2=tf.layers.dense(self.dense1,units=10,name="dense2")

        self.logits=tf.nn.softmax(self.dense2)

        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,labels=self.labels
        ))

        self.accuracy=tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(self.logits,axis=1),tf.argmax(self.labels,1)
                )
                ,tf.float32
            )
        ) * 100

        self.optim=tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(self.loss)

        self.sess=tf.InteractiveSession()
        tf.initialize_all_variables().run()
        return

    def train(self,images,labels):
        '''

        :param images: list of images each of size (28,28)
        :param labels: list of labels each is a one hot vector of size 10
        :return:  loss and accuracy
        '''
        _,lo,acc=self.sess.run([self.optim,self.loss,self.accuracy],feed_dict={
            self.images:images,
            self.labels:labels,
            self.keep_prob:[0.3]
        })

        return lo,acc

    def test(self,images,labels):
        '''

        :param images: list of images each of size (28,28)
        :param labels: list of labels each is a one hot vector of size 10
        :return:  loss and accuracy
        '''
        lo,acc=self.sess.run([self.loss,self.accuracy],feed_dict={
            self.images:images,
            self.labels:labels,
            self.keep_prob:[1]
        })

        return lo,acc
