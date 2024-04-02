import datasets
from .task import register, Task
from .verbsingle import verbsingle


@register('oppoverb_oppo')
class oppoverb_oppo(verbsingle):
    train_files = './data/verb_oppo/oppo_verb_train.json'
    test_files =  './data/verb_oppo/oppo_verb_test.json'

    def doc_to_text(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        inp = f"input: * {a}\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        pair = doc['output'].split(" ")
        a, b = pair
        return a


@register('oppoverb_verb')
class oppoverb_verb(oppoverb_oppo):
    def doc_to_text(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        inp = f"input: # {b}\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        pair = doc['output'].split(" ")
        a, b = pair
        return b

@register('oppoverb_com')
class oppoverb_com(oppoverb_oppo):
    def doc_to_text1(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        inp = f"input: * {a}\noutput: "
        return inp
    
    def doc_to_target1(self, doc):
        pair = doc['output'].split(" ")
        a, b = pair
        return a
    
    def doc_to_text2(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        inp = f"input: # {b}\noutput: "
        return inp
    
    def doc_to_target2(self, doc):
        pair = doc['output'].split(" ")
        a, b = pair
        return b
        
    def doc_to_text(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        inp = f"input: * {a} # {b}\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        oup = doc['output']
        return oup


    def fewshot_context(
        self, doc, num_fewshot, rnd=None, description=None
    ):
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        description = description + "\n\n" if description else ""
        
        if num_fewshot == 0:
            labeled_examples = ""
        else:
            fewshotex = self.fewshot_examples(k = num_fewshot*2, rnd = rnd)

            prompt_list = [
                        self.doc_to_text1(doc) + self.doc_to_target1(doc)
                        for doc in fewshotex[:num_fewshot]
                    ] + \
                    [
                        self.doc_to_text2(doc) + self.doc_to_target2(doc)
                        for doc in fewshotex[num_fewshot:]
                    ]
            
            rnd.shuffle(prompt_list)
            
            # print("fewshotex", fewshotex)
            labeled_examples = (
                "\n\n".join(
                    prompt_list
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)

        return description + labeled_examples + example


@register('oppoverb_com_incontext')
class oppoverb_com_incontext(oppoverb_com):
    def fewshot_context(
        self, doc, num_fewshot, rnd=None, description=None
    ):
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        description = description + "\n\n" if description else ""
        
        if num_fewshot == 0:
            labeled_examples = ""
        else:
            fewshotex = self.fewshot_examples(k = num_fewshot*3, rnd = rnd)

            prompt_list = [
                        self.doc_to_text1(doc) + self.doc_to_target1(doc)
                        for doc in fewshotex[:num_fewshot]
                    ] + \
                    [
                        self.doc_to_text2(doc) + self.doc_to_target2(doc)
                        for doc in fewshotex[num_fewshot:num_fewshot*2]
                    ] + \
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex[num_fewshot*2:]
                    ]
            
            rnd.shuffle(prompt_list)
            
            # print("fewshotex", fewshotex)
            labeled_examples = (
                "\n\n".join(
                    prompt_list
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)

        return description + labeled_examples + example

