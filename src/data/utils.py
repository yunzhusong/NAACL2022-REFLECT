import numpy as np
import pdb
import ast

def doc_truncation(ids=None,
                   masks=None,
                   tokenizer=None,
                   targets=None,
                   targets_ext=None,
                   padding=None,
                   doc_split_id=None,
                   sent_split_id=None,
                   max_source_length=None,
                   max_target_length=None,
                   max_ext_length=None,
                   data_args=None,
                   training_args=None,
                  ):
    """
    Truncate the data in document-wise style
    Args:
        ids: list(list(int)), which haven't do any padding or truncation
        mask: list(list(int)), 0 for mask, 1 for unmask
        targets: list(str), abstraction indices, should be summary
        targets_ext: list(str), extraction indices; should be summary_ext_idx
        doc_split_id: default=2, '<\s>'
        sent_split_id: default=0, '<s>'
    Return:
        model_inputs
    """

    token_id_batch, mask_batch, ext_label_batch = [], [], []
    token_id_rest_batch = []

    extraction_learning = True if targets_ext is not None else False
    abstraction_learning = True if targets is not None else False

    # Process data in batch
    for i, (token_id, mask) in enumerate(zip(ids, masks)):
        token_id = np.array(token_id)
        mask = np.array(mask)

        # Build label for sentence extraction task
        if extraction_learning:

            # Build label for extraction task
            label = np.zeros(len(token_id), dtype=int)
            target = list(map(int, targets_ext[i].split()))
            sent_flags = np.argwhere(token_id==sent_split_id).reshape(-1)
            last_sent = len(sent_flags)

            for t in target:
                sta_flag = sent_flags[t] if t!=0 else 0
                label[sta_flag] = 1

        # Do the document truncation
        if len(token_id) > max_source_length:

            token_id_docs, mask_docs, label_docs = [], [], []
            token_id_docs_rest = []
            doc_sta = 0
            doc_ends = np.argwhere(token_id==doc_split_id).reshape(-1)
            doc_ends = np.append(doc_ends, np.array([len(token_id)-1])) # add the last index

            max_doc_length = int((max_source_length-1)/len(doc_ends))
            for i, doc_end in enumerate(doc_ends):

                doc_len = min(doc_end, doc_sta+max_doc_length)
                token_id_docs.append(token_id[doc_sta:doc_len])
                mask_docs.append(mask[doc_sta:doc_len])

                # Get the rest inputs
                token_id_docs_rest.append(token_id[doc_len:doc_ends[i]])

                if extraction_learning:
                    label_docs.append(label[doc_sta:doc_len]) # For token classification
                
                doc_sta = doc_end

            # Append the end token (<\s>:2)
            token_id_docs.append(token_id[-1:])
            mask_docs.append(mask[-1:])

            # NOTE: Token classification
            label_docs.append(label[-1:] if extraction_learning else None)

            # Concatenate all documents
            token_id = np.concatenate(token_id_docs)
            mask = np.concatenate(mask_docs)
            token_id_rest = np.concatenate(token_id_docs_rest)

            if extraction_learning:
                label = np.concatenate(label_docs)
        else:
            token_id_rest = token_id

        token_id_batch.append(token_id)
        mask_batch.append(mask)
        token_id_rest_batch.append(token_id_rest)

        if extraction_learning:
            position = label[token_id==sent_split_id]
            pad_length = max(0, max_ext_target_length-len(position))
            ext_label_batch.append(np.pad(position[:max_ext_target_length], (0,pad_length), 
                                constant_values=-100))

    if abstraction_learning:
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            label_batch = tokenizer(targets,
                               max_length=max_target_length,
                               padding=padding,
                               truncation=True)["input_ids"]

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
        # when we want to ignore padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            label_batch = [[(l if l != tokenizer.pad_token_id else -100)
                                      for l in label]
                                      for label in label_batch]


        max_length = 3072
        token_id_rest_batch = [np.pad(i[:max_length], (0, max(max_length-len(i),0)), constant_values=tokenizer.pad_token_id) 
                               for i in token_id_rest_batch]


    model_inputs = {}
    model_inputs["input_ids"] = token_id_batch
    model_inputs["attention_mask"] = mask_batch
    #model_inputs["articles"] = token_id_rest_batch

    if abstraction_learning and extraction_learning:
        model_inputs["labels"] = label_batch
        model_inputs["ext_labels"] = ext_label_batch

    elif abstraction_learning:
        model_inputs["labels"] = label_batch

    elif extraction_learning:
        model_inputs["labels"] = ext_label_batch

    return model_inputs
