import json
import os
import re
import numpy as np
import tensorflow as tf

import resources.gpt2.model as model
import resources.gpt2.sample as sample
import resources.gpt2.encoder as encoder


class GPT2:
    def __init__(self, model_path):
        self.model_path = model_path

    def prompt_model(self, prompt,
            seed=None,
            nsamples=1,
            batch_size=1,
            length=None,
            temperature=1,
            top_k=0,
            top_p=0.0
        ):
        ''' With the given GPT Model generates a text given a prompt.
        `seed`=None : Integer seed for random number generators, fix seed to reproduce
        results\n
        `nsamples`=1 : Number of samples to return total\n
        `batch_size`=1 : Number of batches (only affects speed/memory).  Must divide nsamples.\n
        `length`=None : Number of tokens in generated text, if None (default), is
        determined by model hyperparameters\n
        `temperature`=1 : Float value controlling randomness in boltzmann
        distribution. Lower temperature results in less random completions. As the
        temperature approaches zero, the model will become deterministic and
        repetitive. Higher temperature results in more random completions.\n
        `top_k`=0 : Integer value controlling diversity. 1 means only 1 word is
        considered for each step (token), resulting in deterministic completions,
        while 40 means 40 words are considered at each step. 0 (default) is a
        special setting meaning no restrictions. 40 generally is a good value.\n
        `top_p`=0.0 : Float value controlling diversity. Implements nucleus sampling,
        overriding top_k if set to a value > 0. A good setting is 0.9.
        '''
        assert nsamples % batch_size == 0

        model_name = self.model_path

        enc = encoder.get_encoder(model_name)
        hparams = model.default_hparams()
        #with open(os.path.join(model_path, 'hparams.json')) as f:
        with open(os.path.join('models', model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.compat.v1.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, top_p=top_p
            )

            saver = tf.compat.v1.train.Saver()
            #ckpt = tf.train.latest_checkpoint(model_name)
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
            saver.restore(sess, ckpt)

            raw_text = prompt

            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
            print(text)
            text = re.sub(r'^(.|\n)*?Output: ', '', text)
            text = re.sub(r"<\|endtext\|>(.|\s)*", '', text).rstrip(" \r\n")
            
        return text
