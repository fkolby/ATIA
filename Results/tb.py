import tensorboard 
#import tensorflow as tf
import os
import time
cwd = os.getcwd()
logdir = "/home/frederik/mnt/ATIAjbv/ISIC/train/"
logdir2 = cwd + "/Results/Pneumonia/4x4-/lightning_logs/"
print(logdir)

print(os.listdir(logdir))
tb = tensorboard.program.TensorBoard()
pt = 10003
tb.configure(argv = [None, f'--logdir={logdir}', f'--port={pt}'])
tb.launch()


time.sleep(600)