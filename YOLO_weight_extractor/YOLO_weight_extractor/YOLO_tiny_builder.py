import numpy as np
import tensorflow as tf
import cv2

class YOLO_TF:
	weights_dir = 'cjy/'
	out_file = 'model/YOLO_car_tiny.ckpt'
	alpha = 0.1


	def __init__(self):
		self.build_networks()

	def build_networks(self):
		print "Building YOLO_tiny graph..."
		self.x = tf.placeholder('float32',[None,448,448,3])
		self.conv_1 = self.conv_layer(1,self.x,16,3,1)
		self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)
		self.conv_3 = self.conv_layer(3,self.pool_2,32,3,1)
		self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
		self.conv_5 = self.conv_layer(5,self.pool_4,64,3,1)
		self.pool_6 = self.pooling_layer(6,self.conv_5,2,2)
		self.conv_7 = self.conv_layer(7,self.pool_6,128,3,1)
		self.pool_8 = self.pooling_layer(8,self.conv_7,2,2)
		self.conv_9 = self.conv_layer(9,self.pool_8,256,3,1)
		self.pool_10 = self.pooling_layer(10,self.conv_9,2,2)
		self.conv_11 = self.conv_layer(11,self.pool_10,512,3,1)
		self.pool_12 = self.pooling_layer(12,self.conv_11,2,2)
		self.conv_13 = self.conv_layer(13,self.pool_12,1024,3,1)
		self.conv_14 = self.conv_layer(14,self.conv_13,1024,3,1)
		self.conv_15 = self.conv_layer(15,self.conv_14,1024,3,1)
		self.fc_16 = self.fc_layer(16,self.conv_15,256,flat=True,linear=False)
		self.fc_17 = self.fc_layer(17,self.fc_16,4096,flat=False,linear=False)
		#skip dropout_18
		self.fc_19 = self.fc_layer(19,self.fc_17,1331,flat=False,linear=True)
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		self.saver = tf.train.Saver()
		self.saver.save(self.sess,self.out_file)

	def conv_layer(self,idx,inputs,filters,size,stride):
		channels = inputs.get_shape()[3]
		f_w = open(self.weights_dir + str(idx) + '_conv_weights.txt','r')
		l_w = np.array(f_w.readlines()).astype('float32')	
		f_w.close()
		w = np.zeros((size,size,channels,filters),dtype='float32')
		ci = int(channels)
		filter_step = ci*size*size
		channel_step = size*size
		for i in range(filters):
			for j in range(ci):
				for k in range(size):
					for l in range(size):
						w[k,l,j,i] = l_w[i*filter_step + j*channel_step + k*size + l]

		weight = tf.Variable(w,name='yolo/conv_'+str(idx)+'/weights')
		f_b = open(self.weights_dir + str(idx) + '_conv_biases.txt','r')
		l_b = np.array(f_b.readlines()).astype('float32')	
		f_b.close()
		biases = tf.Variable(l_b.reshape((filters)), name='yolo/conv_'+str(idx)+'/biases')

		pad_size = size//2
		pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
		inputs_pad = tf.pad(inputs,pad_mat)

		conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	
		conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
		print 'Loaded ' + str(idx) + ' : conv     from ' + self.weights_dir + str(idx) + '_conv_weights/biases.txt'
		return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

	def pooling_layer(self,idx,inputs,size,stride):
		print 'Create ' + str(idx) + ' : pool'
		return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

	def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
		input_shape = inputs.get_shape().as_list()		
		if flat:
			dim = input_shape[1]*input_shape[2]*input_shape[3]
			inputs_transposed = tf.transpose(inputs,(0,3,1,2))
			inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
		else:
			dim = input_shape[1]
			inputs_processed = inputs
		f_w = open(self.weights_dir + str(idx) + '_fc_weights.txt','r')
		l_w = np.array(f_w.readlines()).astype('float32')
		w = np.zeros((dim,hiddens),dtype='float32')
		for i in range(dim):
			for j in range(hiddens):
				w[i,j] = l_w[j*dim + i]
				#w[i,j]=l_w[i*hiddens+j]
		weight = tf.Variable(w, name='yolo/fc_'+str(idx)+'/weights')
		f_b = open(self.weights_dir + str(idx) + '_fc_biases.txt','r')
		l_b = np.array(f_b.readlines()).astype('float32')		
		biases = tf.Variable(l_b.reshape((hiddens)), name='yolo/fc_'+str(idx)+'/biases')	
		print 'Loaded ' + str(idx) + ' : fc     from ' + self.weights_dir + str(idx) + '_fc_weights/biases.txt'	
		if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
		ip = tf.add(tf.matmul(inputs_processed,weight),biases)
		return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

	def detect_from_cvmat(self,img):
		img_resized = cv2.resize(img, (448, 448))
		img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
		img_resized_np = np.asarray( img_RGB )
		inputs = np.zeros((1,448,448,3),dtype='float32')
		inputs[0] = (img_resized_np/255.0)*2.0-1.0
		in_dict = {self.x: inputs}
		net_output = self.sess.run(self.conv,feed_dict=in_dict)
		for i in range(30):
			print inputs[0,0,i,0]
		print ''
		for i in range(30):
			print net_output[0,0,i,0]
		#f = open("YOLO_output.txt", 'w')
		#net_output.tofile(f,"\n", "%f")

	def detect_from_file(self,filename):
		img = cv2.imread(filename)
		#img = misc.imread(filename)
		self.detect_from_cvmat(img)

	def detect_from_crop_sample(self):
		f = np.array(open('person_crop.txt','r').readlines(),dtype='float32')
		inputs = np.zeros((1,448,448,3),dtype='float32')
		for c in range(3):
			for y in range(448):
				for x in range(448):
					inputs[0,y,x,c] = f[c*448*448+y*448+x]

		#inputs[0] = (img_resized_np/255.0)*2.0-1.0
		in_dict = {self.x: inputs}
		net_output = self.sess.run(self.fc_32,feed_dict=in_dict)
		#for i in range(30):
		#	print inputs[0,0,i,0]
		#print ''
		print net_output.shape
		f = open("YOLO_output.txt", 'w')
		net_output.tofile(f,"\n", "%f")


def main():
	yolo = YOLO_TF()


if __name__=='__main__':
	main()
