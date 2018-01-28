import os
import tqdm
import argparse
import numpy as np
import tensorflow as tf

from torch.utils import data
from tfsemseg.loss import get_loss
from tfsemseg.models import get_model
from tfsemseg.loader import get_data_loader, get_data_path

def train(args):

	data_path = get_data_path(args.dataset)
	loader = get_data_loader(args.dataset)

    tr_loader = loader(data_path, img_size=(args.img_rows, args.img_cols))
    trainloader = data.DataLoader(tr_loader, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, drop_last=True)

    v_loader = loader(data_path, split='val', img_size=(args.img_rows, args.img_cols))
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    n_classes = tr_loader.n_classes

    x = tf.placeholder(shape=[args.batch_size, args.img_rows, args.img_cols, channels=args.channels],dtype=tf.float32)
    y = tf.placeholder(shape=[args.batch_size, args.img_rows, args.img_cols],dtype=tf.int32)

    model get_model(args.arch)
    logits = model(x, n_classes=n_classes, feature_scale=args.feature_scale)

	loss_op = get_loss(args.loss)    
	optimizer = tf.train.AdamOptimizer(learning_rate=args.l_rate).minimize(loss_op)

	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.resume not None:
        	saver.restore(sess, resume)

        best_val_loss = 1.0
        for epoch in range(args.n_epochs):
            tr_loss = 0
            for i, (images, labels) in enumerate(tqdm.tqdm(trainloader)):
                images, labels = images.numpy(), labels.numpy()
                _, loss = sess.run([optimizer, loss_op], feed_dict={x: images, y:labels})
                tr_loss += loss
            tr_loss = tr_loss/(i+1.0)

            val_loss = 0
            for i, (images, labels) in enumerate(tqdm.tqdm(valloader)):
                images, labels = images.numpy(), labels.numpy()
                loss = sess.run(loss_op, feed_dict={x: images, y:labels})
                val_loss += loss
            val_loss = val_loss/(i+1.0)

            print("Epoch (%d/%d), Training Loss: %.3f, Validation Loss: %.3f " % (epoch+1, n_epochs, tr_loss, val_loss))

            if (val_loss < best_val_loss):
                saver.save(sess, save_path)
                print("Model saved in file: %s" % save_path)

    print("Training Finished!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--arch', type=str, default='unet', help='Architecture to use [\'unet, segnet, fcn8s, etc \']')
    parser.add_argument('--dataset', type=str, default='camvid', help='Dataset to use [\'camvid, mit_scene_parser, etc\']')
    parser.add_argument('--n_epochs', type=int, default=101, help='Number of the epochs')
    parser.add_argument('--channels', type=int, default=3, help='Number of the channels')
    parser.add_argument('--img_rows', type=int, default=320, help='Height of the input image')
    parser.add_argument('--img_cols', type=int, default=320, help='Width of the input image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--l_rate', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument('--feature_scale', type=int, default=2, help='Divider for number of features to use')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers to use for data loader')
    parser.add_argument('--loss', type=str, default='dice', help='Loss function to use [\'dice, cross_entropy\']')
    parser.add_argument('--resume', type=str, default=None, help='Path to previous saved model')
    parser.add_argument('--save_dir', type=str, default='savedModel', help='Path to save model')
    args = parser.parse_args()
    train(args)
