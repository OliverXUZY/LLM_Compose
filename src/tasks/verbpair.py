import datasets
from .task import register, Task

@register('verbpair')
class verbpair(Task):
    VERSION = 0
    DATASET_PATH = "json"
    DATASET_NAME = None

    # cache_dir = "./cache"
    train_files = './data/verb_oppo/verb_train.json'
    test_files = './data/verb_oppo/verb_test.json'

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
       
        testset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.test_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        trainset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.train_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        self.dataset = datasets.DatasetDict({
            "train": trainset,
            "validation": testset
        })

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        # inp = f"input: * {a} * {b}\noutput: "
        inp = f"input: * {doc['input']}\noutput: "
        return inp

    def doc_to_target(self, doc):
        return doc["output"]


@register('verbpair_swap')
class verbpair_swap(verbpair):
    def doc_to_text(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        # inp = f"input: {a} # {b} #\noutput: "
        inp = f"input: {doc['input']} #\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        pair = doc['input'].split(" ")
        inp = " ".join(pair)
        a, b = pair
        oup = " ".join([b,a])
        return oup

@register('verbpair_swap_com')
class verbpair_swap_com(verbpair):
    def doc_to_text1(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        # inp = f"input: * {a} * {b}\noutput: "
        inp = f"input: * {doc['input']}\noutput: "
        return inp

    def doc_to_target1(self, doc):
        return doc["output"]
    
    def doc_to_text2(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        # inp = f"input: {a} # {b} #\noutput: "
        inp = f"input: {doc['input']} #\noutput: "
        return inp
    
    def doc_to_target2(self, doc):
        pair = doc['input'].split(" ")
        inp = " ".join(pair)
        a, b = pair
        oup = " ".join([b,a])
        return oup
    
    def doc_to_text(self, doc):
        pair = doc['input'].split(" ")
        a, b = pair
        # inp = f"input: * {a} # * {b} #\noutput: "
        inp = f"input: * {doc['input']} #\noutput: "
        return inp
    
    def doc_to_target(self, doc):
        pair = doc['output'].split(" ")
        inp = " ".join(pair)
        a, b = pair
        oup = " ".join([b,a])
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


@register('verbpair_swap_com_incontext')
class verbpair_swap_com_incontext(verbpair_swap_com):
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
