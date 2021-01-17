import json
import os
import re
import numpy as np
import tensorflow as tf

from .resources.gpt2 import model, sample, encoder


class GPT2:
    def __init__(self, model_path):
        self.model_path = model_path

    def prompt_model(self, prompt,
            temperature=1,
            top_k=0,
            top_p=0.0
        ):
        ''' With the given GPT Model generates a text given a prompt.\n
        `temperature`=1 : Valor controlando la aleatoriedad de la salida\n
        `top_k`=0 : Integer que controla la diversidad. Representa el 
                   número de palabras consideradas. 0 = sin restriccion\n
        `top_p`=0.: Float que controla la diversidad. Un buen valor es 0.9.
        '''
        seed=None
        nsamples=1
        batch_size=1
        length=None

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
            #print(text)
            text = re.sub(r'^(.|\n)*?Output: ', '', text)
            text = re.sub(r"<\|endtext\|>(.|\s)*", '', text).rstrip(" \r\n")
            
        return text
