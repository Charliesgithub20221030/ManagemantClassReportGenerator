from sklearn.metrics.pairwise import cosine_similarity as cos
from tensorflow.compat import v1 as tf
import modeling
import tokenization
from opencc import OpenCC
tf.disable_eager_execution()
cc = OpenCC('t2s')
# # Paremeters
bert_config_file = 'albert_tiny/albert_config_tiny.json'
init_checkpoint = "albert_tiny/albert_model.ckpt"
is_training = False
is_eval = False
do_lower_case = True
vocab_file = 'albert_tiny/vocab.txt'
max_seq_length = 32

label_list = ['0', '1', '2']
num_labels = len(label_list)
compute_loss = False
# input_ids = ?  # input_ids: int32 Tensor of shape [batch_size, seq_length]


# # [from input_fn ]
# #  name_to_features = {
# #         "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
# #         "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
# #         "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
# #         "label_ids": tf.FixedLenFeature([], tf.int64),
# #         "is_real_example": tf.FixedLenFeature([], tf.int64),
# #     }

# num_labels = len(label_list)
# learning_rate = 0
# num_train_steps = 0
# num_warmup_steps = 0
# use_tpu = False
# use_one_hot_embeddings = False


# train_examples = processor.get_train_examples(FLAGS.data_dir)  # TODO
# print("###length of total train_examples:", len(train_examples))
# num_train_steps = int(len(train_examples) /
#                       FLAGS.train_batch_size * FLAGS.num_train_epochs)
# num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


bert_config = modeling.BertConfig.from_json_file(bert_config_file)
# init_checkpoint_path = ''


"""
is_training:
bool. true for training model, false for eval model. Controls
whether dropout will be applied.
"""

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=do_lower_case)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data.
    inpuy_ids: token 對應編號

    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_example(example, label_list, max_seq_length):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def create_examples(line):
    """ ex. line:0,i love you,i hate you """
    line = line.split(',')
    set_type = 'test'
    try:
        guid = "%s-%s" % (set_type, 0)
        label = tokenization.convert_to_unicode(line[0])
        text_a = tokenization.convert_to_unicode(line[1])
        if len(line) > 2:
            text_b = tokenization.convert_to_unicode(line[2])
        else:
            text_b = None

        example = [
            InputExample(guid=guid, text_a=text_a,
                         text_b=text_b, label=label)
        ]
    except Exception as e:
        print(e)
    return example


def get_output(line='0,你好,我很好'):
    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.

    line = cc.convert(line)
    tf.reset_default_graph()
    examples = create_examples(line)
    feature = convert_single_example(example=examples[0],
                                     label_list=label_list,
                                     max_seq_length=max_seq_length)
    print("input feature: \nids:%s\nsegment ids%s\nmask%s" %
          (feature.input_ids, feature.segment_ids, feature.input_mask))
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=tf.convert_to_tensor([feature.input_ids], dtype=tf.int32, name="input_ids"))

    # print(f"beta test :\n{model.run_model()}\n")

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1]

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        ln_type = bert_config.ln_type
        if ln_type == 'preln':  # add by brightmart, 10-06. if it is preln, we need to an additonal layer: layer normalization as suggested in paper "ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE"
            print("ln_type is preln. add LN layer.")
            output_layer = layer_norm(output_layer)
        else:
            print("ln_type is postln or other,do nothing.")

        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        # output logits
        logits = tf.matmul(
            output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        # prob from output logits
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        loss = None
        per_example_loss = None
        # labels are needed
        if compute_loss:
            one_hot_labels = tf.one_hot(
                labels, depth=num_labels, dtype=tf.float32)

            # todo 08-29 try temp-loss
            per_example_loss = - \
                tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            ###############bi_tempered_logistic_loss############################################################################
            # print("##cross entropy loss is used...."); tf.logging.info("##cross entropy loss is used....")
            # t1=0.9 #t1=0.90
            # t2=1.05 #t2=1.05
            # per_example_loss=bi_tempered_logistic_loss(log_probs,one_hot_labels,t1,t2,label_smoothing=0.1,num_iters=5) # TODO label_smoothing=0.0
            # tf.logging.info("per_example_loss:"+str(per_example_loss.shape))
            ##############bi_tempered_logistic_loss#############################################################################

            loss = tf.reduce_mean(per_example_loss)

        # with tf.Session() as sess:
        #     with sess.as_default():
        #         tf.global_variables_initializer().run()
        #         sess.run(logits)
        #         print(logits.eval())

        """ important information from create_model: (not initialized)
         (loss, per_example_loss, logits, probabilities)"""

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if is_training:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif is_eval:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(
                    values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            # test block
            # we just want to get embedding
            #################################################
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     predictions={"probabilities": probabilities},
            #     scaffold_fn=scaffold_fn)
            ###################################################

            with tf.Session() as sess:
                with sess.as_default():
                    tf.global_variables_initializer().run()
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    output_logits = sess.run(logits)
                    output_hidden = sess.run(model.get_pooled_output())
                    # print(logits.eval())

        # return output_spec
        return [output_logits, output_hidden]


# (loss, per_example_loss, logits, probabilities) = get_output()
logit1, hidden1 = get_output('0,不')
logit2, hidden2 = get_output('0,摩羯座')
logit3, hidden3 = get_output('0,天蠍座')

print("1 to 2", cos(hidden1, hidden2))
print('2 to 3', cos(hidden2, hidden3))
print('1 to 3', cos(hidden1, hidden3))
