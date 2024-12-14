import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt ############## added for plotting
from codebase.metrics import DI, DP, DI_soft
from codebase.results import ResultLogger
from codebase.tester import Tester

# defaults
BATCH_SIZE = 32
AUD_STEPS = 1
BIG = 100000
CLASS_BOUND = 1.8
DISC_BOUND = 1.8
NUM_CONSEC_DISC_BOUND_0 = 10
NUM_CONSEC_NO_TRAIN_CLASS = 10


class Trainer(object):
    def __init__(self,
                 model,
                 data,
                 batch_size=32,
                 learning_rate=0.001,
                 sess=None,
                 logs_path='./tfboard_logs',
                 expdir=None,
                 regbas=False,  # regbas (originally 'regularized baseline') means we are training a naive
                                # (or fair-regularized) classifier from pre-trained representations (e.g., a previous
                                # LAFTR run), i.e., not training LAFTR from scratch
                 aud_steps=AUD_STEPS,
                 **kwargs):
        self.data = data
        if not self.data.loaded:
            self.data.load()
            self.data.make_validation_set()
        self.model = model
        self.batch_size = batch_size
        self.batches_seen = 0
        self.logs_path = logs_path
        self.expdir = expdir
        self.regbas = regbas
        self.aud_steps = aud_steps

        ################## get the NEW flag from a file ##################
        file_data = []
        with open('new.txt') as f:
            for num in f:
                file_data.append(int(num))
        self.NEW = file_data[0] == 1
        self.INDEX = file_data[1]
        ###############################################################

        # encoder-classifier-decoder train op
        self.opt_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.enc_cla_op = self.opt_gen.minimize(
            self.model.loss,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/enc_cla')
        )

        # auditor train op
        self.opt_aud = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.aud_op = self.opt_aud.minimize(
            -self.model.loss,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/aud')
        )

        # get enc-cla-dec gradients
        gen_grads = self.opt_gen.compute_gradients(
            self.model.loss,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/enc_cla')
        )
        self.gen_grads = list(filter(lambda g: not g[0] is None and not 'reconstructed' in g[1].name, gen_grads))

        # TODO: actual summaries here
        # for g in gen_grads:
        #     tf.summary.histogram('gen_grads_{}'.format(g[1].name), g[0])

        # get aud gradients
        aud_grads = self.opt_aud.compute_gradients(
            -self.model.loss,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/aud')
        )
        self.aud_grads = list(filter(lambda g: not g[0] is None and not 'reconstructed' in g[1].name, aud_grads))
        # for g in aud_grads
        #     tf.summary.histogram('aud_grads_{}'.format(g[1].name), g[0])

        ########################################################## added for plotting
        # Modifications for ploting:
        # Initialize lists to store metrics
        self.di_values = []
        self.dp_values = []
        self.train_class_loss = []
        self.train_disc_loss = []
        self.train_total_loss = []
        self.valid_class_loss = []
        self.valid_disc_loss = []
        self.valid_total_loss = []
        self.train_class_err = []
        self.train_aud_err = []
        self.valid_class_err = []
        self.valid_aud_err = []
        if self.NEW: # if new experiment, initialize list to store DXC values (fairness metric)
            self.DXC = []
        ################################################

        self.summ_op = self.make_summaries()
        self.sess = sess or tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # for saving model checkpoints

    def plot_metrics(self): ################# new function to plot the training losses and metrics
        # Plot DI and DP in single figure window with two axes
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].plot(self.di_values, 'r-', label='DI')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('DI', color='r')
        axs[0].legend()

        axs[1].plot(self.dp_values, 'b-', label='DP')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('DP', color='b')
        axs[1].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(self.expdir, 'DI_DP_over_epochs.png'))
        plt.close()

        # Plot training and validation losses in a new figure with six axes
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].plot(self.train_class_loss, label='ClaCE (Train)')
        axs[0, 0].set_title('ClaCE (Train)')
        axs[0, 0].set(xlabel='Epoch', ylabel='Loss')
        axs[0, 0].legend()

        axs[1, 0].plot(self.valid_class_loss, label='ClaCE (Valid)')
        axs[1, 0].set_title('ClaCE (Valid)')
        axs[1, 0].set(xlabel='Epoch', ylabel='Loss')
        axs[1, 0].legend()

        axs[0, 1].plot(self.train_disc_loss, label='DisCE (Train)')
        axs[0, 1].set_title('DisCE (Train)')
        axs[0, 1].set(xlabel='Epoch', ylabel='Loss')
        axs[0, 1].legend()

        axs[1, 1].plot(self.valid_disc_loss, label='DisCE (Valid)')
        axs[1, 1].set_title('DisCE (Valid)')
        axs[1, 1].set(xlabel='Epoch', ylabel='Loss')
        axs[1, 1].legend()

        axs[0, 2].plot(self.train_total_loss, label='TtlCE (Train)')
        axs[0, 2].set_title('TtlCE (Train)')
        axs[0, 2].set(xlabel='Epoch', ylabel='Loss')
        axs[0, 2].legend()

        axs[1, 2].plot(self.valid_total_loss, label='TtlCE (Valid)')
        axs[1, 2].set_title('TtlCE (Valid)')
        axs[1, 2].set(xlabel='Epoch', ylabel='Loss')
        axs[1, 2].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(self.expdir, 'Train_Valid_Losses.png'))
        plt.close()

        # Plot training and validation errors in a new figure with four axes
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(self.train_class_err, label='ClaErr (Train)')
        axs[0, 0].set_title('ClaErr (Train)')
        axs[0, 0].set(xlabel='Epoch', ylabel='Error')
        axs[0, 0].legend()

        axs[1, 0].plot(self.valid_class_err, label='ClaErr (Valid)')
        axs[1, 0].set_title('ClaErr (Valid)')
        axs[1, 0].set(xlabel='Epoch', ylabel='Error')
        axs[1, 0].legend()

        axs[0, 1].plot(self.train_aud_err, label='aud_err (Train)')
        axs[0, 1].set_title('aud_err (Train)')
        axs[0, 1].set(xlabel='Epoch', ylabel='Error')
        axs[0, 1].legend()

        axs[1, 1].plot(self.valid_aud_err, label='aud_err (Valid)')
        axs[1, 1].set_title('aud_err (Valid)')
        axs[1, 1].set(xlabel='Epoch', ylabel='Error')
        axs[1, 1].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(self.expdir, 'Train_Valid_Errors.png'))
        plt.close()

        if self.NEW:
            fig, axs = plt.subplots(1, 2, figsize=(15, 10))
            axs[0].plot(self.DXC, 'b-', label='DXC')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('DXC', color='b')
            axs[0].legend()

            axs[1].plot(self.dp_values, 'b-', label='DP')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('DP', color='b')
            axs[1].legend()

            fig.tight_layout()
            plt.savefig(os.path.join(self.expdir, 'DXC_DP_over_epochs.png'))
            plt.close()

        ######################################## Save metrics to text files
        with open(os.path.join(self.expdir,"di_values.txt"), "w") as f:
            for s in self.di_values:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"dp_values.txt"), "w") as f:
            for s in self.dp_values:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"train_class_loss.txt"), "w") as f:
            for s in self.train_class_loss:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"train_disc_loss.txt"), "w") as f:
            for s in self.train_disc_loss:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"train_total_loss.txt"), "w") as f:
            for s in self.train_total_loss:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"valid_class_loss.txt"), "w") as f:
            for s in self.valid_class_loss:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"valid_disc_loss.txt"), "w") as f:
            for s in self.valid_disc_loss:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"valid_total_loss.txt"), "w") as f:
            for s in self.valid_total_loss:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"train_class_err.txt"), "w") as f:
            for s in self.train_class_err:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"train_aud_err.txt"), "w") as f:
            for s in self.train_aud_err:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"valid_class_err.txt"), "w") as f:
            for s in self.valid_class_err:
                f.write(str(s) + "\n")
        with open(os.path.join(self.expdir,"valid_aud_err.txt"), "w") as f:
            for s in self.valid_aud_err:
                f.write(str(s) + "\n")
        if self.NEW:
            with open(os.path.join(self.expdir, "DXC.txt"), "w") as f:
                for s in self.DXC:
                    f.write(str(s) + "\n")


        ##########################################


    def make_summaries(self):  # TODO: fill me in
        tf.summary.histogram('Ahat', self.model.A_hat)
        return tf.summary.merge_all()

    ################## trainNEW is the new training function that is used for the new experiment ##################
    ################## trainNEW saves the metrics to text files and plots the metrics ##################
    ################## This function consider the existance of Xc as input to the model, recieved from the get_batch_iterator #################
    ################## This function also calcuate DXC as a fairness metric ##################
    def trainNEW(self, n_epochs, patience, **kwargs):
        summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)

        min_val_loss, min_epoch = np.finfo(np.float32).max, -1

        class_dp_last_ep = BIG
        disc_dp_bound_last_ep = BIG

        for epoch in range(n_epochs):
            print('starting Epoch {:d}'.format(epoch))
            train_iter = self.data.get_batch_iterator('train', self.batch_size)
            train_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0.}
            self.batches_seen = 0
            trained_class = 0; trained_aud = 0
            Y_hats_tr = np.empty((0, 1))
            A_hats_tr = np.empty((0, 1))

            print('Class DP last epoch: {:.3f}; Disc DP Bound last epoch: {:.3f}'.format(class_dp_last_ep, disc_dp_bound_last_ep))
            trained_class_this_epoch = False
            trained_disc_this_epoch = False

            for x, y, a, xc in train_iter: ################## added xc to the iterator
                if len(x) < self.batch_size:  # hack for WGAN-GP training; don't process weird-sized batches
                    continue

                ################### Xc is input also to the model ###################
                feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a,self.model.XC: xc, self.model.epoch: np.array([epoch])}

                if self.batches_seen == 0:  # a quick summary before any training
                    summary_writer.add_summary(self.sess.run(self.summ_op, feed_dict=feed_dict), epoch)
                self.batches_seen += 1
                # train encoder-classifier-decoder
                _, class_loss, class_err, recon_loss, Y_hat, A_hat = self.sess.run(
                    [self.enc_cla_op,
                     self.model.class_loss,
                     self.model.class_err,
                     self.model.recon_loss,
                     self.model.Y_hat,
                     self.model.A_hat
                     ],
                    feed_dict=feed_dict
                )
                trained_class_this_epoch = True
                trained_class += 1

                aud_ops_base = [self.model.loss, self.model.aud_loss, self.model.aud_err, \
                                        self.model.Y_hat, self.model.A_hat]

                for _ in range(self.aud_steps):
                    if not self.regbas:
                        trained_disc_this_epoch = True
                        trained_aud += 1
                        aud_ops = [self.aud_op] + aud_ops_base
                        # train auditor
                        _, total_loss, aud_loss, aud_err, Y_hat, A_hat = self.sess.run(
                                aud_ops,
                                feed_dict=feed_dict
                                )
                    else:
                        # train auditor
                        total_loss, aud_loss, aud_err, Y_hat, A_hat = self.sess.run(
                            aud_ops_base,
                            feed_dict=feed_dict
                        )

                Y_hats_tr = np.concatenate((Y_hats_tr, Y_hat))
                A_hats_tr = np.concatenate((A_hats_tr, A_hat))

                # TODO rename for clarity and consistency
                train_L['class'] += np.mean(class_loss)
                train_L['disc'] += np.mean(aud_loss)
                train_L['class_err'] += class_err
                train_L['disc_err'] += aud_err
                train_L['recon'] += np.mean(recon_loss)

            print('E{:d}: trained class {:d}, trained aud {:d}'.format(epoch, trained_class, trained_aud))
            for k in train_L:
                train_L[k] /= self.batches_seen
            train_L['ttl'] = train_L['class'] - train_L['disc']
            train_res_str = 'E{:d}: ClaCE:{:.3f}, DisCE:{:.3f}, TtlCE:{:.3f}, ClaErr:{:.3f}, DisErr:{:.3f}, RecLoss:{:.3f}'.\
                            format(epoch, train_L['class'], train_L['disc'], train_L['ttl'], train_L['class_err'], train_L['disc_err'], train_L['recon'])

            ################################ For plotting function
            self.train_class_loss.append(train_L['class'])
            self.train_disc_loss.append(train_L['disc'])
            self.train_total_loss.append(train_L['ttl'])
            self.train_class_err.append(train_L['class_err'])
            self.train_aud_err.append(train_L['disc_err'])
            ################################


            valid_iter = self.data.get_batch_iterator('valid', self.batch_size)
            valid_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0., 'baseline_aud': 0., 'final_aud': 0.}
            num_batches = 0
            Y_hats = np.empty((0, 1))
            Ys = np.empty((0, 1))
            As = np.empty((0, 1))
            A_hats = np.empty((0, 1))
            if self.NEW:
                Xc = np.empty((0, 1)) ################## added for DXC calculation

            for x, y, a, xc in valid_iter: ################## added xc to the iterator
                num_batches += 1
                if len(x) < self.batch_size:  # hack for WGAN-GP training; don't process weird-sized batches
                    continue

                feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a,
                             self.model.XC: xc, self.model.epoch: np.array([epoch])} ############### Xc is input also to the model


                #  run encoder-classifier-decoder (don't take a train step)
                class_loss, class_err, recon_loss, Y_hat, A_hat, total_loss, aud_loss, aud_err = self.sess.run(
                    [self.model.class_loss,
                     self.model.class_err,
                     self.model.recon_loss,
                     self.model.Y_hat,
                     self.model.A_hat,
                     self.model.loss,
                     self.model.aud_loss,
                     self.model.aud_err],
                    feed_dict=feed_dict
                )

                # TODO rename for clarity and consistency
                valid_L['class'] += np.mean(class_loss)
                valid_L['disc'] += np.mean(aud_loss)
                valid_L['class_err'] += class_err
                valid_L['disc_err'] += aud_err
                valid_L['recon'] += np.mean(recon_loss)

                Y_hats = np.concatenate((Y_hats, Y_hat))
                Ys = np.concatenate((Ys, y))
                As = np.concatenate((As, a))
                A_hats = np.concatenate((A_hats, A_hat))

                Xc = np.concatenate((Xc, xc)) ################## added for DXC calculation

                if hasattr(self.model,'baseline_aud_loss'):
                    baseline_aud_loss, final_aud_loss = self.sess.run(
                        [self.model.baseline_aud_loss, self.model.final_aud_loss],
                        feed_dict=feed_dict
                    )
                    valid_L['baseline_aud'] += np.mean(baseline_aud_loss)
                    valid_L['final_aud'] += np.mean(final_aud_loss)

            for k in valid_L:
                valid_L[k] /= num_batches
            valid_L['ttl'] = valid_L['class'] - valid_L['disc']
            valid_res_str = 'E{:d}: ClaCE:{:.3f}, DisCE:{:.3f}, TtlCE:{:.3f}, ClaErr:{:.3f}, DisErr:{:.3f}, RecLoss:{:.3f}'.\
                format(epoch, valid_L['class'], valid_L['disc'], valid_L['ttl'], valid_L['class_err'], valid_L['disc_err'], valid_L['recon'])

            ################################ ################# For plotting function
            self.valid_class_loss.append(valid_L['class'])
            self.valid_disc_loss.append(valid_L['disc'])
            self.valid_total_loss.append(valid_L['ttl'])
            self.valid_class_err.append(valid_L['class_err'])
            self.valid_aud_err.append(valid_L['disc_err'])
            ################################


            # Create a new Summary object with your measure
            summary = tf.Summary()
            #summary.value.add(tag="class_loss", simple_value=valid_L['class'])
            summary.value.add(tag="scaled_class_loss", simple_value=self.model.class_coeff*valid_L['class'])
            #summary.value.add(tag="disc_loss", simple_value=valid_L['disc'])
            summary.value.add(tag="scaled_disc_loss", simple_value=self.model.fair_coeff*valid_L['disc'])
            summary.value.add(tag="class_err", simple_value=valid_L['class_err'])
            summary.value.add(tag="disc_err", simple_value=valid_L['disc_err'])

            di = DI(Ys, Y_hats, As) * 2
            print('DI: ', di)
            summary.value.add(tag="DI", simple_value=di)
            demo_dispar = DP(Y_hats, As)
            summary.value.add(tag="DP", simple_value=demo_dispar)
            print('DP: ', demo_dispar)

            if self.NEW: ################# added for DXC calculation
                dxc = DI(Xc, Y_hats, As)
                print('DXC: ', dxc)
                summary.value.add(tag="DXC", simple_value=dxc)
                self.DXC.append(dxc)
            #######################
            self.di_values.append(di)
            self.dp_values.append(demo_dispar)
            ######################


            if epoch % 50 == 0 and not self.regbas:
                # Valid set
                # create a new folder to log in
                new_dname_v = os.path.join(self.expdir, 'checkpoints', 'Epoch_{:d}_Valid'.format(epoch))
                # create reslogger for that folder
                reslogger_v = ResultLogger(new_dname_v, self.saver)
                # create tester for that reslogger
                tester_v = Tester(self.model, self.data, self.sess, reslogger_v)
                # run tester.evaluate
                tester_v.evaluateNEW(self.batch_size, phase='valid', save=not self.regbas) ################# changed to evaluateNEW

                # Test set
                # create a new folder to log in
                new_dname = os.path.join(self.expdir, 'checkpoints', 'Epoch_{:d}_Test'.format(epoch))
                # create reslogger for that folder
                reslogger = ResultLogger(new_dname, self.saver)
                # create tester for that reslogger
                tester = Tester(self.model, self.data, self.sess, reslogger)
                # run tester.evaluate
                tester.evaluateNEW(self.batch_size, phase='test', save=not self.regbas)

            if not self.regbas:
                from codebase.models import WeightedDemParWassGpGan
                if isinstance(self.model, WeightedDemParWassGpGan):
                    summary.value.add(tag="grad_norm_err", simple_value=self.sess.run(self.model.grad_norms, feed_dict=feed_dict))

            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()

            if epoch % 1 == 0:
                print('{}; {}'.format(train_res_str, valid_res_str))

            l = valid_L['class'] if self.regbas else valid_L['ttl']

            if l < min_val_loss:
                min_val_loss = l
                min_epoch = epoch
                if self.regbas:
                    test_iter = self.data.get_batch_iterator('test', self.batch_size)
                    test_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0}
                    num_batches = 0

                    for x, y, a, xc in test_iter:
                        num_batches += 1



                        # make feed dict
                        feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a,
                                 self.model.XC: xc,
                                 self.model.epoch: np.array([epoch])}

                        # feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a,
                        #              self.model.epoch: np.array([epoch])}

                        # class_loss, class_err, Y_hat, Z, Y = self.sess.run(
                        class_loss, recon_loss, class_err, Y_hat, Z, Y, XC= self.sess.run(
                            [self.model.class_loss,
                             self.model.recon_loss,
                             self.model.class_err,
                             self.model.Y_hat,
                             self.model.Z,
                             self.model.Y,
                             self.model.XC], ################# added for Xc is added
                            feed_dict=feed_dict
                        )
                        # train discriminator
                        aud_loss, aud_err, total_loss, A_hat, A, XC = self.sess.run(
                            [self.model.aud_loss,
                            self.model.aud_err,
                            self.model.loss,
                            self.model.A_hat,
                            self.model.A,
                            self.model.XC], ################# added for Xc is added
                            feed_dict=feed_dict)

                        test_L['class'] += np.mean(class_loss)
                        test_L['disc'] += np.mean(aud_loss)
                        test_L['class_err'] += class_err
                        test_L['disc_err'] += aud_err
                        test_L['recon'] += np.mean(recon_loss)

                    for k in test_L:
                        test_L[k] /= num_batches

                    test_L['ttl'] = test_L['class'] - test_L['disc']
                    test_res_str = 'Test score: Class CE: {class:.3f}, Disc CE: {disc:.3f}, Ttl CE: {ttl:.3f},' + \
                                   ' Class Err: {class_err:.3f} Disc Err: {disc_err:.3f}'
                    print(test_res_str.format(**test_L))

            if epoch == n_epochs - 1 or epoch - min_epoch >= patience:  # stop training if the loss isn't going down
                print('Finished training: min validation loss was {:.3f} in epoch {:d}'.format(min_val_loss, min_epoch))
                break
        self.plot_metrics()
        with open(os.path.join(self.expdir,"min_epoch.text"), "w") as f:
            f.write(str(min_epoch))
        return

    ########### For this function is the original training function that is used for the original experiment ######
    ########### We added only the plotting function to this function ######
    ########### filling the lists for plots is added to this function ######
    def train(self, n_epochs, patience, **kwargs):
        summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)

        min_val_loss, min_epoch = np.finfo(np.float32).max, -1

        class_dp_last_ep = BIG
        disc_dp_bound_last_ep = BIG

        for epoch in range(n_epochs):
            print('starting Epoch {:d}'.format(epoch))
            train_iter = self.data.get_batch_iterator('train', self.batch_size)
            train_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0.}
            self.batches_seen = 0
            trained_class = 0; trained_aud = 0
            Y_hats_tr = np.empty((0, 1))
            A_hats_tr = np.empty((0, 1))

            print('Class DP last epoch: {:.3f}; Disc DP Bound last epoch: {:.3f}'.format(class_dp_last_ep, disc_dp_bound_last_ep))
            trained_class_this_epoch = False
            trained_disc_this_epoch = False

            for x, y, a in train_iter:
                if len(x) < self.batch_size:  # hack for WGAN-GP training; don't process weird-sized batches
                    continue
                feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a, self.model.epoch: np.array([epoch])}
                if self.batches_seen == 0:  # a quick summary before any training
                    summary_writer.add_summary(self.sess.run(self.summ_op, feed_dict=feed_dict), epoch)
                self.batches_seen += 1
                # train encoder-classifier-decoder
                _, class_loss, class_err, recon_loss, Y_hat, A_hat = self.sess.run(
                    [self.enc_cla_op,
                     self.model.class_loss,
                     self.model.class_err,
                     self.model.recon_loss,
                     self.model.Y_hat,
                     self.model.A_hat
                     ],
                    feed_dict=feed_dict
                )
                trained_class_this_epoch = True
                trained_class += 1

                aud_ops_base = [self.model.loss, self.model.aud_loss, self.model.aud_err, \
                                        self.model.Y_hat, self.model.A_hat]

                for _ in range(self.aud_steps):
                    if not self.regbas:
                        trained_disc_this_epoch = True
                        trained_aud += 1
                        aud_ops = [self.aud_op] + aud_ops_base
                        # train auditor
                        _, total_loss, aud_loss, aud_err, Y_hat, A_hat = self.sess.run(
                                aud_ops,
                                feed_dict=feed_dict
                                )
                    else:
                        # train auditor
                        total_loss, aud_loss, aud_err, Y_hat, A_hat = self.sess.run(
                            aud_ops_base,
                            feed_dict=feed_dict
                        )

                Y_hats_tr = np.concatenate((Y_hats_tr, Y_hat))
                A_hats_tr = np.concatenate((A_hats_tr, A_hat))

                # TODO rename for clarity and consistency
                train_L['class'] += np.mean(class_loss)
                train_L['disc'] += np.mean(aud_loss)
                train_L['class_err'] += class_err
                train_L['disc_err'] += aud_err
                train_L['recon'] += np.mean(recon_loss)

            print('E{:d}: trained class {:d}, trained aud {:d}'.format(epoch, trained_class, trained_aud))
            for k in train_L:
                train_L[k] /= self.batches_seen
            train_L['ttl'] = train_L['class'] - train_L['disc']
            train_res_str = 'E{:d}: ClaCE:{:.3f}, DisCE:{:.3f}, TtlCE:{:.3f}, ClaErr:{:.3f}, DisErr:{:.3f}, RecLoss:{:.3f}'.\
                            format(epoch, train_L['class'], train_L['disc'], train_L['ttl'], train_L['class_err'], train_L['disc_err'], train_L['recon'])

            ################################
            self.train_class_loss.append(train_L['class'])
            self.train_disc_loss.append(train_L['disc'])
            self.train_total_loss.append(train_L['ttl'])
            self.train_class_err.append(train_L['class_err'])
            self.train_aud_err.append(train_L['disc_err'])
            ################################


            valid_iter = self.data.get_batch_iterator('valid', self.batch_size)
            valid_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0., 'baseline_aud': 0., 'final_aud': 0.}
            num_batches = 0
            Y_hats = np.empty((0, 1))
            Ys = np.empty((0, 1))
            As = np.empty((0, 1))
            A_hats = np.empty((0, 1))

            for x, y, a in valid_iter:
                num_batches += 1
                if len(x) < self.batch_size:  # hack for WGAN-GP training; don't process weird-sized batches
                    continue
                feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a, self.model.epoch: np.array([epoch])}
                #  run encoder-classifier-decoder (don't take a train step)
                class_loss, class_err, recon_loss, Y_hat, A_hat, total_loss, aud_loss, aud_err = self.sess.run(
                    [self.model.class_loss,
                     self.model.class_err,
                     self.model.recon_loss,
                     self.model.Y_hat,
                     self.model.A_hat,
                     self.model.loss,
                     self.model.aud_loss,
                     self.model.aud_err],
                    feed_dict=feed_dict
                )

                # TODO rename for clarity and consistency
                valid_L['class'] += np.mean(class_loss)
                valid_L['disc'] += np.mean(aud_loss)
                valid_L['class_err'] += class_err
                valid_L['disc_err'] += aud_err
                valid_L['recon'] += np.mean(recon_loss)

                Y_hats = np.concatenate((Y_hats, Y_hat))
                Ys = np.concatenate((Ys, y))
                As = np.concatenate((As, a))
                A_hats = np.concatenate((A_hats, A_hat))

                if hasattr(self.model, 'baseline_aud_loss'):
                    baseline_aud_loss, final_aud_loss = self.sess.run(
                        [self.model.baseline_aud_loss,
                         self.model.final_aud_loss],
                        feed_dict=feed_dict
                    )
                    valid_L['baseline_aud'] += np.mean(baseline_aud_loss)
                    valid_L['final_aud'] += np.mean(final_aud_loss)

            for k in valid_L:
                valid_L[k] /= num_batches
            valid_L['ttl'] = valid_L['class'] - valid_L['disc']
            valid_res_str = 'E{:d}: ClaCE:{:.3f}, DisCE:{:.3f}, TtlCE:{:.3f}, ClaErr:{:.3f}, DisErr:{:.3f}, RecLoss:{:.3f}'.\
                format(epoch, valid_L['class'], valid_L['disc'], valid_L['ttl'], valid_L['class_err'], valid_L['disc_err'], valid_L['recon'])

            ################################
            self.valid_class_loss.append(valid_L['class'])
            self.valid_disc_loss.append(valid_L['disc'])
            self.valid_total_loss.append(valid_L['ttl'])
            self.valid_class_err.append(valid_L['class_err'])
            self.valid_aud_err.append(valid_L['disc_err'])
            ################################


            # Create a new Summary object with your measure
            summary = tf.Summary()
            #summary.value.add(tag="class_loss", simple_value=valid_L['class'])
            summary.value.add(tag="scaled_class_loss", simple_value=self.model.class_coeff*valid_L['class'])
            #summary.value.add(tag="disc_loss", simple_value=valid_L['disc'])
            summary.value.add(tag="scaled_disc_loss", simple_value=self.model.fair_coeff*valid_L['disc'])
            summary.value.add(tag="class_err", simple_value=valid_L['class_err'])
            summary.value.add(tag="disc_err", simple_value=valid_L['disc_err'])

            di = DI(Ys, Y_hats, As) * 2
            print('DI: ', di)
            summary.value.add(tag="DI", simple_value=di)
            demo_dispar = DP(Y_hats, As)
            summary.value.add(tag="DP", simple_value=demo_dispar)
            print('DP: ', demo_dispar)

            #######################
            self.di_values.append(di)
            self.dp_values.append(demo_dispar)
            ######################


            if epoch % 50 == 0 and not self.regbas:
                # Valid set
                # create a new folder to log in
                new_dname_v = os.path.join(self.expdir, 'checkpoints', 'Epoch_{:d}_Valid'.format(epoch))
                # create reslogger for that folder
                reslogger_v = ResultLogger(new_dname_v, self.saver)
                # create tester for that reslogger
                tester_v = Tester(self.model, self.data, self.sess, reslogger_v)
                # run tester.evaluate
                tester_v.evaluate(self.batch_size, phase='valid', save=not self.regbas)

                # Test set
                # create a new folder to log in
                new_dname = os.path.join(self.expdir, 'checkpoints', 'Epoch_{:d}_Test'.format(epoch))
                # create reslogger for that folder
                reslogger = ResultLogger(new_dname, self.saver)
                # create tester for that reslogger
                tester = Tester(self.model, self.data, self.sess, reslogger)
                # run tester.evaluate
                tester.evaluate(self.batch_size, phase='test', save=not self.regbas)

            if not self.regbas:
                from codebase.models import WeightedDemParWassGpGan
                if isinstance(self.model, WeightedDemParWassGpGan):
                    summary.value.add(tag="grad_norm_err", simple_value=self.sess.run(self.model.grad_norms, feed_dict=feed_dict))

            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()

            if epoch % 1 == 0:
                print('{}; {}'.format(train_res_str, valid_res_str))

            l = valid_L['class'] if self.regbas else valid_L['ttl']

            if l < min_val_loss:
                min_val_loss = l
                min_epoch = epoch
                if self.regbas:
                    test_iter = self.data.get_batch_iterator('test', self.batch_size)
                    test_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0}
                    num_batches = 0

                    for x, y, a in test_iter:
                        num_batches += 1

                        # make feed dict
                        feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a,
                                     self.model.epoch: np.array([epoch])}

                        # class_loss, class_err, Y_hat, Z, Y = self.sess.run(
                        class_loss, recon_loss, class_err, Y_hat, Z, Y = self.sess.run(
                            [self.model.class_loss,
                             self.model.recon_loss,
                             self.model.class_err,
                             self.model.Y_hat,
                             self.model.Z,
                             self.model.Y],
                            feed_dict=feed_dict
                        )
                        # train discriminator
                        aud_loss, aud_err, total_loss, A_hat, A = self.sess.run(
                            [self.model.aud_loss,
                             self.model.aud_err,
                             self.model.loss,
                             self.model.A_hat,
                             self.model.A],
                            feed_dict=feed_dict)

                        test_L['class'] += np.mean(class_loss)
                        test_L['disc'] += np.mean(aud_loss)
                        test_L['class_err'] += class_err
                        test_L['disc_err'] += aud_err
                        test_L['recon'] += np.mean(recon_loss)

                    for k in test_L:
                        test_L[k] /= num_batches

                    test_L['ttl'] = test_L['class'] - test_L['disc']
                    test_res_str = 'Test score: Class CE: {class:.3f}, Disc CE: {disc:.3f}, Ttl CE: {ttl:.3f},' + \
                                   ' Class Err: {class_err:.3f} Disc Err: {disc_err:.3f}'
                    print(test_res_str.format(**test_L))

            if epoch == n_epochs - 1 or epoch - min_epoch >= patience:  # stop training if the loss isn't going down
                print('Finished training: min validation loss was {:.3f} in epoch {:d}'.format(min_val_loss, min_epoch))
                break
        self.plot_metrics()
        with open(os.path.join(self.expdir,"min_epoch.text"), "w") as f:
            f.write(str(min_epoch))
        return

    ################# This function is not used
    def train22(self, n_epochs, patience, **kwargs):
        summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)

        min_val_loss, min_epoch = np.finfo(np.float32).max, -1

        class_dp_last_ep = BIG
        disc_dp_bound_last_ep = BIG

        for epoch in range(n_epochs):
            print('starting Epoch {:d}'.format(epoch))
            train_iter = self.data.get_batch_iterator('train', self.batch_size)
            train_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0.}
            self.batches_seen = 0
            trained_class = 0;
            trained_aud = 0
            Y_hats_tr = np.empty((0, 1))
            A_hats_tr = np.empty((0, 1))

            print('Class DP last epoch: {:.3f}; Disc DP Bound last epoch: {:.3f}'.format(class_dp_last_ep,
                                                                                         disc_dp_bound_last_ep))
            trained_class_this_epoch = False
            trained_disc_this_epoch = False

            for x, y, a in train_iter:
                if len(x) < self.batch_size:  # hack for WGAN-GP training; don't process weird-sized batches
                    continue
                feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a, self.model.epoch: np.array([epoch])}
                if self.batches_seen == 0:  # a quick summary before any training
                    summary_writer.add_summary(self.sess.run(self.summ_op, feed_dict=feed_dict), epoch)
                self.batches_seen += 1
                # train encoder-classifier-decoder
                _, class_loss, class_err, recon_loss, Y_hat, A_hat = self.sess.run(
                    [self.enc_cla_op,
                     self.model.class_loss,
                     self.model.class_err,
                     self.model.recon_loss,
                     self.model.Y_hat,
                     self.model.A_hat
                     ],
                    feed_dict=feed_dict
                )
                trained_class_this_epoch = True
                trained_class += 1

                aud_ops_base = [self.model.loss, self.model.aud_loss, self.model.aud_err, \
                                self.model.Y_hat, self.model.A_hat]

                for _ in range(self.aud_steps):
                    if not self.regbas:
                        trained_disc_this_epoch = True
                        trained_aud += 1
                        aud_ops = [self.aud_op] + aud_ops_base
                        # train auditor
                        _, total_loss, aud_loss, aud_err, Y_hat, A_hat = self.sess.run(
                            aud_ops,
                            feed_dict=feed_dict
                        )
                    else:
                        # train auditor
                        total_loss, aud_loss, aud_err, Y_hat, A_hat = self.sess.run(
                            aud_ops_base,
                            feed_dict=feed_dict
                        )

                Y_hats_tr = np.concatenate((Y_hats_tr, Y_hat))
                A_hats_tr = np.concatenate((A_hats_tr, A_hat))

                # TODO rename for clarity and consistency
                train_L['class'] += np.mean(class_loss)
                train_L['disc'] += np.mean(aud_loss)
                train_L['class_err'] += class_err
                train_L['disc_err'] += aud_err
                train_L['recon'] += np.mean(recon_loss)

            print('E{:d}: trained class {:d}, trained aud {:d}'.format(epoch, trained_class, trained_aud))
            for k in train_L:
                train_L[k] /= self.batches_seen
            train_L['ttl'] = train_L['class'] - train_L['disc']
            train_res_str = 'E{:d}: ClaCE:{:.3f}, DisCE:{:.3f}, TtlCE:{:.3f}, ClaErr:{:.3f}, DisErr:{:.3f}, RecLoss:{:.3f}'. \
                format(epoch, train_L['class'], train_L['disc'], train_L['ttl'], train_L['class_err'],
                       train_L['disc_err'], train_L['recon'])

            valid_iter = self.data.get_batch_iterator('valid', self.batch_size)
            valid_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0., 'baseline_aud': 0.,
                       'final_aud': 0.}
            num_batches = 0
            Y_hats = np.empty((0, 1))
            Ys = np.empty((0, 1))
            As = np.empty((0, 1))
            A_hats = np.empty((0, 1))

            for x, y, a in valid_iter:
                num_batches += 1
                if len(x) < self.batch_size:  # hack for WGAN-GP training; don't process weird-sized batches
                    continue
                feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a, self.model.epoch: np.array([epoch])}
                #  run encoder-classifier-decoder (don't take a train step)
                class_loss, class_err, recon_loss, Y_hat, A_hat, total_loss, aud_loss, aud_err = self.sess.run(
                    [self.model.class_loss,
                     self.model.class_err,
                     self.model.recon_loss,
                     self.model.Y_hat,
                     self.model.A_hat,
                     self.model.loss,
                     self.model.aud_loss,
                     self.model.aud_err],
                    feed_dict=feed_dict
                )

                # TODO rename for clarity and consistency
                valid_L['class'] += np.mean(class_loss)
                valid_L['disc'] += np.mean(aud_loss)
                valid_L['class_err'] += class_err
                valid_L['disc_err'] += aud_err
                valid_L['recon'] += np.mean(recon_loss)

                Y_hats = np.concatenate((Y_hats, Y_hat))
                Ys = np.concatenate((Ys, y))
                As = np.concatenate((As, a))
                A_hats = np.concatenate((A_hats, A_hat))

                if hasattr(self.model, 'baseline_aud_loss'):
                    baseline_aud_loss, final_aud_loss = self.sess.run(
                        [self.model.baseline_aud_loss,
                         self.model.final_aud_loss],
                        feed_dict=feed_dict
                    )
                    valid_L['baseline_aud'] += np.mean(baseline_aud_loss)
                    valid_L['final_aud'] += np.mean(final_aud_loss)

            for k in valid_L:
                valid_L[k] /= num_batches
            valid_L['ttl'] = valid_L['class'] - valid_L['disc']
            valid_res_str = 'E{:d}: ClaCE:{:.3f}, DisCE:{:.3f}, TtlCE:{:.3f}, ClaErr:{:.3f}, DisErr:{:.3f}, RecLoss:{:.3f}'. \
                format(epoch, valid_L['class'], valid_L['disc'], valid_L['ttl'], valid_L['class_err'],
                       valid_L['disc_err'], valid_L['recon'])

            # Create a new Summary object with your measure
            summary = tf.Summary()
            # summary.value.add(tag="class_loss", simple_value=valid_L['class'])
            summary.value.add(tag="scaled_class_loss", simple_value=self.model.class_coeff * valid_L['class'])
            # summary.value.add(tag="disc_loss", simple_value=valid_L['disc'])
            summary.value.add(tag="scaled_disc_loss", simple_value=self.model.fair_coeff * valid_L['disc'])
            summary.value.add(tag="class_err", simple_value=valid_L['class_err'])
            summary.value.add(tag="disc_err", simple_value=valid_L['disc_err'])

            di = DI(Ys, Y_hats, As) * 2
            print('DI: ', di)
            summary.value.add(tag="DI", simple_value=di)
            demo_dispar = DP(Y_hats, As)
            summary.value.add(tag="DP", simple_value=demo_dispar)
            print('DP: ', demo_dispar)

            if epoch % 50 == 0 and not self.regbas:
                # Valid set
                # create a new folder to log in
                new_dname_v = os.path.join(self.expdir, 'checkpoints', 'Epoch_{:d}_Valid'.format(epoch))
                # create reslogger for that folder
                reslogger_v = ResultLogger(new_dname_v, self.saver)
                # create tester for that reslogger
                tester_v = Tester(self.model, self.data, self.sess, reslogger_v)
                # run tester.evaluate
                tester_v.evaluate(self.batch_size, phase='valid', save=not self.regbas)

                # Test set
                # create a new folder to log in
                new_dname = os.path.join(self.expdir, 'checkpoints', 'Epoch_{:d}_Test'.format(epoch))
                # create reslogger for that folder
                reslogger = ResultLogger(new_dname, self.saver)
                # create tester for that reslogger
                tester = Tester(self.model, self.data, self.sess, reslogger)
                # run tester.evaluate
                tester.evaluate(self.batch_size, phase='test', save=not self.regbas)

            if not self.regbas:
                from codebase.models import WeightedDemParWassGpGan
                if isinstance(self.model, WeightedDemParWassGpGan):
                    summary.value.add(tag="grad_norm_err",
                                      simple_value=self.sess.run(self.model.grad_norms, feed_dict=feed_dict))

            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()

            if epoch % 1 == 0:
                print('{}; {}'.format(train_res_str, valid_res_str))

            l = valid_L['class'] if self.regbas else valid_L['ttl']

            if l < min_val_loss:
                min_val_loss = l
                min_epoch = epoch
                if self.regbas:
                    test_iter = self.data.get_batch_iterator('test', self.batch_size)
                    test_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0}
                    num_batches = 0

                    for x, y, a in test_iter:
                        num_batches += 1

                        # make feed dict
                        feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a,
                                     self.model.epoch: np.array([epoch])}

                        # class_loss, class_err, Y_hat, Z, Y = self.sess.run(
                        class_loss, recon_loss, class_err, Y_hat, Z, Y = self.sess.run(
                            [self.model.class_loss,
                             self.model.recon_loss,
                             self.model.class_err,
                             self.model.Y_hat,
                             self.model.Z,
                             self.model.Y],
                            feed_dict=feed_dict
                        )
                        # train discriminator
                        aud_loss, aud_err, total_loss, A_hat, A = self.sess.run(
                            [self.model.aud_loss,
                             self.model.aud_err,
                             self.model.loss,
                             self.model.A_hat,
                             self.model.A],
                            feed_dict=feed_dict)

                        test_L['class'] += np.mean(class_loss)
                        test_L['disc'] += np.mean(aud_loss)
                        test_L['class_err'] += class_err
                        test_L['disc_err'] += aud_err
                        test_L['recon'] += np.mean(recon_loss)

                    for k in test_L:
                        test_L[k] /= num_batches

                    test_L['ttl'] = test_L['class'] - test_L['disc']
                    test_res_str = 'Test score: Class CE: {class:.3f}, Disc CE: {disc:.3f}, Ttl CE: {ttl:.3f},' + \
                                   ' Class Err: {class_err:.3f} Disc Err: {disc_err:.3f}'
                    print(test_res_str.format(**test_L))

            if epoch == n_epochs - 1 or epoch - min_epoch >= patience:  # stop training if the loss isn't going down
                print('Finished training: min validation loss was {:.3f} in epoch {:d}'.format(min_val_loss, min_epoch))
                break
        return