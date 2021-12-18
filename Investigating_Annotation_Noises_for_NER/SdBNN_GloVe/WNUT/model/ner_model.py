import numpy as np
import os
import tensorflow as tf
import time


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.trans_params = tf.get_variable("crf_trans_params", dtype=tf.float32,
                            shape=[self.config.ntags, self.config.ntags])



    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.max_seq_len],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.config.batch_size],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.max_seq_len, None],
                        name="char_ids")
        self.softmax_pc = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.config.max_seq_len, self.config.ntags],
                        name="soft_max_pc")

        self.softmax_pm = tf.placeholder(tf.float32,
                                          shape=[self.config.batch_size, self.config.max_seq_len, self.config.ntags],
                                          name="soft_max_pm")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.max_seq_len],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.max_seq_len],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None,softmax_pc = None,softmax_pm = None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids,self.config.max_seq_len, 0)
            char_ids, word_lengths = pad_sequences(char_ids,self.config.max_seq_len, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, self.config.max_seq_len,0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels,self.config.max_seq_len, 0)
            feed[self.labels] = labels

        if softmax_pm is not None:
            softmax_pm,_ = pad_sequences(softmax_pm,self.config.max_seq_len, 0.,nlevels=3)
            feed[self.softmax_pm] = softmax_pm

        if softmax_pc is not None:
            softmax_pc,_ = pad_sequences(softmax_pc,self.config.max_seq_len, 0.,nlevels=3)
            feed[self.softmax_pc] = softmax_pc

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)
                print("word_embedding shape is: ",word_embeddings.shape)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            print(output.shape)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            W_var = tf.get_variable("W_var", dtype=tf.float32,
                                    shape=[2 * self.config.hidden_size_lstm, 1])

            b_var = tf.get_variable("b_var", shape=[1],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
            W_var1 = tf.get_variable("W_var1", dtype=tf.float32,
                                     shape=[self.config.batch_size, self.config.max_seq_len, self.config.ntags])
            W_var2 = tf.get_variable("W_var2", dtype=tf.float32,
                                     shape=[self.config.batch_size, self.config.max_seq_len, self.config.ntags])

            #b_var2 = tf.get_variable("b_var2", shape=[self.config.batch_size, self.config.max_seq_len, self.config.ntags],
                                    #dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            print("nsteps: ",nsteps)
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            var_pred = tf.matmul(output, W_var) + b_var
            self.vars = tf.keras.layers.Activation('softplus', name='variance')(var_pred)
            self.vars = tf.reshape(self.vars, [-1, nsteps, 1])
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])
            self.logits_vars = tf.concat([self.logits, self.vars], axis=-1)
            self.system_vars = W_var1*self.softmax_pc+W_var2*self.softmax_pm

    def standard_crf_loss(self, pred):
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            pred, self.labels, self.sequence_lengths, transition_params=self.trans_params)
        # mask = tf.sequence_mask(self.sequence_lengths)
        # losses = tf.boolean_mask(losses, mask)
        # loss = tf.reshape(losses,[-1])
        losses = -log_likelihood
        return losses
        # loss = tf.reduce_mean(losses)

    def gaussian_categorical_crossentropy(self, dist, undistorted_loss):
        def map_fn(i):
            std_sample = tf.transpose(dist.sample(self.config.ntags), [1, 2, 0])
            # print("shape of trans_std_sample is: ",std_sample.shape)
            pred = self.logits + std_sample+self.system_vars
            # [batch_size,seq_len]
            distored_loss = self.standard_crf_loss(pred)
            diff = undistorted_loss - distored_loss
            return -tf.keras.backend.elu(diff)

        return map_fn

    def data_uncertainty_loss(self, T):
        std = tf.sqrt(self.logits_vars[:, :, self.config.ntags])
        # print("shape of std is: ", std.shape)
        dist = tf.distributions.Normal(loc=tf.zeros_like(std), scale=std)
        undistorted_loss = self.standard_crf_loss(self.logits)
        iterable = tf.keras.backend.variable(np.ones(T))
        monte_carlo_results = tf.keras.backend.map_fn(
            self.gaussian_categorical_crossentropy(dist, undistorted_loss), iterable, name='monte_carlo_results')
        variance_loss = tf.keras.backend.mean(monte_carlo_results, axis=0) * undistorted_loss
        variance_depressor = tf.keras.backend.exp(self.vars) - tf.keras.backend.ones_like(self.vars)
        variance_depressor = tf.reduce_mean(variance_depressor)
        uncertainly_loss = variance_loss + undistorted_loss
        uncertainly_loss = tf.reduce_mean(uncertainly_loss)+variance_depressor
        softmax_loss = tf.reduce_mean(undistorted_loss)
        self.loss =  0.5*uncertainly_loss + softmax_loss


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            self.data_uncertainty_loss(100)
        else:
            self.data_uncertainty_loss(100)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)
            # print("the shape of logits is : ",logits)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                # print("one of logit shape is: ",logit.shape)
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                # print("pick max one is: ",viterbi_seq)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev,test, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        start_time = time.time()
        # print("original train: ",len(train))
        # padding_train_num = len(train) % 20
        # print(padding_train_num)
        # print("padding train: ",train)
        #prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels,softmax_pc,softmax_pm) in enumerate(minibatches(train, batch_size)):
            batch_start_time = time.time()
            fd, _ = self.get_feed_dict(words, labels,softmax_pc,softmax_pm, self.config.lr,
                    self.config.dropout)
            # print(fd)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            #prog.update(i + 1, [("train loss", train_loss)])
            batch_time = time.time()
            if i % batch_size == 0:
                self.logger.info("training batch: %5d, time consumption:%.2f(s),loss: %.5f, " % (
                i, batch_time - batch_start_time, train_loss))
                
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        test_metrics = self.run_evaluate(test)
        test_msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in test_metrics.items()])
        end_time = time.time()
        self.logger.info('\ntime consumption:%.2f(min), deving results is :%s, testing results is : %s' %((end_time-start_time)/60,msg,test_msg))

        return test_metrics["f1"] #why test


    def run_evaluate(self, test):
    
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels,softmax_pc,softmax_pm in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1, "percision":100*p, "recall":100*r}
        
    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
