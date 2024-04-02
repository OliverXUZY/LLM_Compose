
import datasets
from .task import register, Task
from .verbpair import verbpair

@register('verbsingle')
class verbsingle(verbpair):
    train_files = './data/verb_oppo/verb_single_train.json'
    test_files = './data/verb_oppo/verb_single_test.json'

    def doc_to_text(self, doc):
        inp = f"input: * {doc['input']}\noutput: "
        return inp
    
@register('verbsingle_upper')
class verbsingle_upper(verbsingle):
    def doc_to_text(self, doc):
        inp = f"input: {doc['input']} #\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        return doc['input'].upper()
    
@register('verbsingle_upper_com')
class verbsingle_upper_com(verbsingle):
    def doc_to_text1(self, doc):
        inp = f"input: * {doc['input']}\noutput: "
        return inp

    def doc_to_target1(self, doc):
        return doc["output"]
    
    def doc_to_text2(self, doc):
        inp = f"input: {doc['input']} #\noutput: "
        return inp
    
    def doc_to_target2(self, doc):
        return doc['input'].upper()
    
    def doc_to_text(self, doc):
        inp = f"input: * {doc['input']} #\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        return doc['output'].upper()

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


@register('verbsingle_upper_com_incontext')
class verbsingle_upper_com_incontext(verbsingle_upper_com):
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





### plusOne
import hashlib
import random

def deterministic_random(doc_id):
    # Convert the doc_id to a string and encode it to bytes
    doc_id_str = str(doc_id).encode('utf-8')
    
    # Use a hash function to generate a hash value from the doc_id
    hash_value = hashlib.sha256(doc_id_str).hexdigest()
    
    # Convert the first few characters of the hash to an integer
    # This serves as a deterministic seed value based on the doc_id
    seed_value = int(hash_value[:8], 16)
    
    # Use the seed to initialize the random number generator
    random.seed(seed_value)
    
    # Generate and return a random number between 1 and 1000
    return random.randint(1, 1000)

@register('verbsingle_word')
class verbsingle_word(verbpair):
    train_files = './data/verb_oppo/verb_single_train.json'
    test_files = './data/verb_oppo/verb_single_test.json'

    def doc_to_text(self, doc):
        inp = f"input:{doc['input']}\noutput: "
        return inp

@register('verbsingle_plusOne')
class verbsingle_plusOne(verbsingle):
    def doc_to_text(self, doc):
        num = deterministic_random(doc['input'])
        inp = f"input: {num}\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        num = deterministic_random(doc['input'])
        return str(num + 1)
    
@register('verbsingle_plusOne_com')
class verbsingle_plusOne_com(verbsingle_plusOne):
    def doc_to_text1(self, doc):
        inp = f"input: {doc['input']}\noutput: "
        return inp

    def doc_to_target1(self, doc):
        return doc["output"]
    
    def doc_to_text2(self, doc):
        num = deterministic_random(doc['input'])
        inp = f"input: {num}\noutput: "
        return inp
    
    def doc_to_target2(self, doc):
        num = deterministic_random(doc['input'])
        return str(num + 1)
       
    def doc_to_text(self, doc):
        num = deterministic_random(doc['input'])
        inp = f"input: {num} {doc['input']}\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        num = deterministic_random(doc['input'])
        oup =f"{num + 1} {doc['output']}"
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


@register('verbsingle_plusOne_com_incontext')
class verbsingle_plusOne_com_incontext(verbsingle_plusOne_com):
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


